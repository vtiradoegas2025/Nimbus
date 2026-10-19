use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::error::{AppError, AppResult};

const MAX_NAMELIST_BYTES: u64 = 8 * 1024 * 1024;
const MAX_ASSIGNMENTS: usize = 100_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Cm1NamelistAssignment {
    pub key: String,
    pub raw_value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Cm1NamelistGroup {
    pub name: String,
    pub assignments: Vec<Cm1NamelistAssignment>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Cm1NamelistReport {
    pub schema: &'static str,
    pub schema_version: u32,
    pub path: String,
    pub groups: Vec<Cm1NamelistGroup>,
    pub assignment_count: usize,
    pub values: BTreeMap<String, String>,
    pub warnings: Vec<String>,
}

#[tauri::command]
pub async fn inspect_cm1_namelist(path: String) -> AppResult<Cm1NamelistReport> {
    tauri::async_runtime::spawn_blocking(move || {
        let metadata = std::fs::metadata(&path)?;
        if metadata.len() > MAX_NAMELIST_BYTES {
            return Err(AppError::Invalid(
                "CM1 namelist exceeds the 8 MiB inspection safety limit".into(),
            ));
        }
        let contents = std::fs::read_to_string(&path).map_err(|error| {
            AppError::Invalid(format!("cannot read CM1 namelist as UTF-8 text: {error}"))
        })?;
        parse_cm1_namelist(&contents, path)
    })
    .await
    .map_err(|error| AppError::Task(error.to_string()))?
}

fn strip_comment(line: &str) -> String {
    let mut quote = None;
    let mut result = String::new();
    let mut chars = line.chars().peekable();
    while let Some(character) = chars.next() {
        if let Some(active) = quote {
            result.push(character);
            if character == active {
                if chars.peek() == Some(&active) {
                    result.push(chars.next().unwrap());
                } else {
                    quote = None;
                }
            }
        } else if character == '\'' || character == '"' {
            quote = Some(character);
            result.push(character);
        } else if character == '!' {
            break;
        } else {
            result.push(character);
        }
    }
    result
}

fn terminator_index(value: &str) -> Option<usize> {
    let mut quote = None;
    let mut characters = value.char_indices().peekable();
    while let Some((index, character)) = characters.next() {
        if let Some(active) = quote {
            if character == active {
                if characters.peek().map(|(_, next)| *next) == Some(active) {
                    characters.next();
                } else {
                    quote = None;
                }
            }
        } else if character == '\'' || character == '"' {
            quote = Some(character);
        } else if character == '/' {
            return Some(index);
        }
    }
    None
}

fn assignment_starts(contents: &str) -> Vec<(usize, usize, String)> {
    let bytes = contents.as_bytes();
    let mut starts = Vec::new();
    let mut index = 0;
    let mut quote = None;
    while index < bytes.len() {
        let character = bytes[index] as char;
        if let Some(active) = quote {
            if character == active {
                if index + 1 < bytes.len() && bytes[index + 1] as char == active {
                    index += 2;
                    continue;
                }
                quote = None;
            }
            index += 1;
            continue;
        }
        if character == '\'' || character == '"' {
            quote = Some(character);
            index += 1;
            continue;
        }
        if !(character.is_ascii_alphabetic() || character == '_') {
            index += 1;
            continue;
        }
        let key_start = index;
        index += 1;
        while index < bytes.len() {
            let current = bytes[index] as char;
            if current.is_ascii_alphanumeric() || matches!(current, '_' | '%') {
                index += 1;
            } else {
                break;
            }
        }
        let mut cursor = index;
        while cursor < bytes.len() && (bytes[cursor] as char).is_ascii_whitespace() {
            cursor += 1;
        }
        if cursor < bytes.len() && bytes[cursor] == b'(' {
            let mut depth = 0_i32;
            while cursor < bytes.len() {
                match bytes[cursor] {
                    b'(' => depth += 1,
                    b')' => {
                        depth -= 1;
                        if depth == 0 {
                            cursor += 1;
                            break;
                        }
                    }
                    _ => {}
                }
                cursor += 1;
            }
            while cursor < bytes.len() && (bytes[cursor] as char).is_ascii_whitespace() {
                cursor += 1;
            }
        }
        if cursor < bytes.len() && bytes[cursor] == b'=' {
            let key = contents[key_start..cursor]
                .chars()
                .filter(|character| !character.is_ascii_whitespace())
                .collect::<String>()
                .to_ascii_lowercase();
            starts.push((key_start, cursor + 1, key));
            index = cursor + 1;
        }
    }
    starts
}

fn parse_assignments(contents: &str) -> Vec<Cm1NamelistAssignment> {
    let starts = assignment_starts(contents);
    starts
        .iter()
        .enumerate()
        .map(|(index, (_, value_start, key))| {
            let value_end = starts
                .get(index + 1)
                .map(|value| value.0)
                .unwrap_or(contents.len());
            let raw_value = contents[*value_start..value_end]
                .trim()
                .trim_end_matches(',')
                .trim()
                .to_string();
            Cm1NamelistAssignment {
                key: key.clone(),
                raw_value,
            }
        })
        .collect()
}

fn parse_cm1_namelist(contents: &str, path: String) -> AppResult<Cm1NamelistReport> {
    let mut groups = Vec::new();
    let mut warnings = Vec::new();
    let mut active_name: Option<String> = None;
    let mut active_contents = String::new();

    for (line_index, original) in contents.lines().enumerate() {
        let without_comment = strip_comment(original);
        let trimmed = without_comment.trim();
        if active_name.is_none() {
            if let Some(rest) = trimmed
                .strip_prefix('&')
                .or_else(|| trimmed.strip_prefix('$'))
            {
                let name_end = rest
                    .find(|character: char| character.is_ascii_whitespace() || character == ',')
                    .unwrap_or(rest.len());
                let name = rest[..name_end].trim().to_ascii_lowercase();
                if name.is_empty() {
                    warnings.push(format!(
                        "line {} has an empty namelist group name",
                        line_index + 1
                    ));
                    continue;
                }
                active_name = Some(name);
                let remainder = rest[name_end..].trim_start_matches(',').trim();
                if let Some(end) = terminator_index(remainder) {
                    active_contents.push_str(&remainder[..end]);
                    groups.push(Cm1NamelistGroup {
                        name: active_name.take().unwrap(),
                        assignments: parse_assignments(&active_contents),
                    });
                    active_contents.clear();
                } else {
                    active_contents.push_str(remainder);
                    active_contents.push('\n');
                }
            }
            continue;
        }
        if trimmed.eq_ignore_ascii_case("$end") || trimmed.eq_ignore_ascii_case("&end") {
            groups.push(Cm1NamelistGroup {
                name: active_name.take().unwrap(),
                assignments: parse_assignments(&active_contents),
            });
            active_contents.clear();
        } else if let Some(end) = terminator_index(&without_comment) {
            active_contents.push_str(&without_comment[..end]);
            groups.push(Cm1NamelistGroup {
                name: active_name.take().unwrap(),
                assignments: parse_assignments(&active_contents),
            });
            active_contents.clear();
        } else {
            active_contents.push_str(&without_comment);
            active_contents.push('\n');
        }
    }
    if let Some(name) = active_name.take() {
        warnings.push(format!("namelist group '&{name}' is not terminated"));
        groups.push(Cm1NamelistGroup {
            name,
            assignments: parse_assignments(&active_contents),
        });
    }

    let assignment_count: usize = groups.iter().map(|group| group.assignments.len()).sum();
    if assignment_count > MAX_ASSIGNMENTS {
        return Err(AppError::Invalid(
            "CM1 namelist exceeds the 100,000-assignment inspection safety limit".into(),
        ));
    }
    let mut values = BTreeMap::new();
    let mut seen = BTreeSet::new();
    for group in &groups {
        for assignment in &group.assignments {
            let identity = format!("{}.{}", group.name, assignment.key);
            if !seen.insert(identity.clone()) {
                warnings.push(format!(
                    "duplicate assignment '{identity}'; the last raw value is shown"
                ));
            }
            values.insert(identity, assignment.raw_value.clone());
        }
    }
    if groups.is_empty() {
        warnings.push("no Fortran namelist groups were found".into());
    }
    Ok(Cm1NamelistReport {
        schema: "nimbus.studio.cm1-namelist-report",
        schema_version: 1,
        path,
        groups,
        assignment_count,
        values,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_groups_comments_quotes_arrays_and_duplicates_without_translation() {
        let report = parse_cm1_namelist(
            "! case\n&param0 nx=64, ny = 32, title='storm!case', arr(1)=3*0 /\n&param1\n run_time=1800.0, dtl=2.0 ! seconds\n dtl=1.5\n$end\n",
            "namelist.input".into(),
        )
        .unwrap();
        assert_eq!(report.groups.len(), 2);
        assert_eq!(
            report.values.get("param0.nx").map(String::as_str),
            Some("64")
        );
        assert_eq!(
            report.values.get("param0.title").map(String::as_str),
            Some("'storm!case'")
        );
        assert_eq!(
            report.values.get("param0.arr(1)").map(String::as_str),
            Some("3*0")
        );
        assert_eq!(
            report.values.get("param1.dtl").map(String::as_str),
            Some("1.5")
        );
        assert!(report
            .warnings
            .iter()
            .any(|warning| warning.contains("duplicate assignment")));
    }

    #[test]
    fn preserves_unterminated_groups_with_an_explicit_warning() {
        let report = parse_cm1_namelist("&param0 nx=10", "namelist.input".into()).unwrap();
        assert_eq!(
            report.values.get("param0.nx").map(String::as_str),
            Some("10")
        );
        assert!(report
            .warnings
            .iter()
            .any(|warning| warning.contains("not terminated")));
    }

    #[test]
    fn ignores_slashes_inside_fortran_strings_with_doubled_quotes() {
        let report = parse_cm1_namelist(
            "&param0 title='don''t / stop', nx=10 /",
            "namelist.input".into(),
        )
        .unwrap();
        assert_eq!(
            report.values.get("param0.title").map(String::as_str),
            Some("'don''t / stop'")
        );
        assert_eq!(
            report.values.get("param0.nx").map(String::as_str),
            Some("10")
        );
        assert!(report.warnings.is_empty());
    }
}
