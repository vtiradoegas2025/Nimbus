use std::collections::BTreeSet;

use serde_json::Value;

use crate::error::{AppError, AppResult};

const STATES: [&str; 5] = [
    "engineering_incomplete",
    "ready",
    "campaign_validated",
    "externally_reviewed",
    "fully_qualified",
];

fn required_string<'a>(item: &'a Value, field: &str) -> AppResult<&'a str> {
    item.get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| AppError::Subprocess(format!("readiness item has no valid {field}")))
}

pub(crate) fn validate_readiness_contract(catalog: &Value) -> AppResult<()> {
    let readiness = catalog
        .get("readiness")
        .and_then(Value::as_object)
        .ok_or_else(|| AppError::Subprocess("scheme catalog has no readiness contract".into()))?;
    if readiness.get("version").and_then(Value::as_u64) != Some(1) {
        return Err(AppError::Subprocess(
            "scheme catalog has an unsupported readiness contract".into(),
        ));
    }
    let items = readiness
        .get("items")
        .and_then(Value::as_array)
        .filter(|items| !items.is_empty())
        .ok_or_else(|| AppError::Subprocess("scheme catalog has no readiness items".into()))?;
    let mut ids = BTreeSet::new();
    for item in items {
        let id = required_string(item, "id")?;
        let area = required_string(item, "area")?;
        let scope = required_string(item, "scope")?;
        let state = required_string(item, "state")?;
        let bounded = required_string(item, "boundedEvidence")?;
        let campaign = required_string(item, "campaignEvidence")?;
        let review = required_string(item, "externalReviewEvidence")?;
        let blocker = required_string(item, "blocker")?;
        if id.is_empty()
            || id != id.trim().to_lowercase()
            || !ids.insert(id)
            || area.trim().is_empty()
            || scope.trim().is_empty()
            || !STATES.contains(&state)
        {
            return Err(AppError::Subprocess(
                "scheme catalog has a malformed readiness item".into(),
            ));
        }
        let rank = STATES
            .iter()
            .position(|candidate| candidate == &state)
            .expect("state membership checked above");
        if rank == 0 {
            if blocker.trim().is_empty() {
                return Err(AppError::Subprocess(format!(
                    "readiness item '{id}' has no engineering blocker"
                )));
            }
        } else if !blocker.trim().is_empty() || bounded.trim().is_empty() {
            return Err(AppError::Subprocess(format!(
                "readiness item '{id}' contradicts its ready state"
            )));
        }
        if rank >= 2 && campaign.trim().is_empty() {
            return Err(AppError::Subprocess(format!(
                "readiness item '{id}' has no campaign evidence"
            )));
        }
        if rank >= 3 && review.trim().is_empty() {
            return Err(AppError::Subprocess(format!(
                "readiness item '{id}' has no external review evidence"
            )));
        }
    }
    Ok(())
}

pub(crate) fn validate_scheme_readiness(catalog: &Value) -> AppResult<()> {
    let modules = catalog
        .get("modules")
        .and_then(Value::as_array)
        .ok_or_else(|| AppError::Subprocess("scheme catalog modules are malformed".into()))?;
    for module in modules {
        let module_id = module
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("<unknown>");
        let schemes = module
            .get("schemes")
            .and_then(Value::as_array)
            .ok_or_else(|| AppError::Subprocess("scheme catalog schemes are malformed".into()))?;
        for scheme in schemes {
            let scheme_id = scheme
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("<unknown>");
            let state = scheme
                .get("readinessState")
                .and_then(Value::as_str)
                .filter(|state| STATES.contains(state))
                .ok_or_else(|| {
                    AppError::Subprocess(format!(
                        "scheme '{module_id}.{scheme_id}' has no valid readiness state"
                    ))
                })?;
            if state != "engineering_incomplete"
                && (scheme.get("constructible").and_then(Value::as_bool) != Some(true)
                    || scheme.get("runtimeSelectable").and_then(Value::as_bool) != Some(true)
                    || scheme.get("maturity").and_then(Value::as_str) != Some("qualified"))
            {
                return Err(AppError::Subprocess(format!(
                    "scheme '{module_id}.{scheme_id}' contradicts its readiness state"
                )));
            }
        }
    }
    Ok(())
}
