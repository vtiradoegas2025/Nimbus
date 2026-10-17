use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde::Serialize;

/*
 * In-memory state for a running Studio session.
 *
 * Each run is held behind its own `Arc<Mutex<RunRecord>>`. This lets the
 * three threads we spawn per run (stdout reader, stderr reader, waiter)
 * each hold a clone and mutate the record without colliding with the
 * command handlers that snapshot it for the UI.
 */

#[derive(Debug)]
pub struct RunRecord {
    pub id: String,
    pub case_name: String,
    /// Dynamics scheme captured from the config at spawn (e.g.
    /// "supercell_cgrid"). Empty when the caller didn't supply one.
    pub scheme: String,
    /// Configured run length in seconds (v1 `time.run_time`). 0 when the
    /// caller didn't supply one. Used by the UI progress-bar denominator.
    pub duration_s: f64,
    /// Source YAML path. Surfaced in Phase 1.E run-detail view.
    #[allow(dead_code)]
    pub config_path: PathBuf,
    pub pid: u32,
    pub status: RunStatus,
    /// Set by `abort_sim` when the user requests termination. The status stays
    /// `Running` until the waiter observes the child actually exit, at which
    /// point this flag tells the waiter to record the run as `Aborted` rather
    /// than Completed/Failed. Keeping it separate from `status` is what stops
    /// the gallery from showing "Aborted" while the process is still alive.
    pub abort_requested: bool,
    pub progress: Option<RunProgress>,
    /// Wall-clock timestamp (unix ms) when spawn returned. Used by the UI
    /// to compute elapsed_s without round-tripping a new field.
    pub started_at_ms: u128,
    /// Wall-clock timestamp (unix ms) when the waiter observed the child
    /// exit. Set once and never updated; lets the UI freeze the wall
    /// clock on completed/failed/aborted runs instead of letting it tick
    /// forever against `now()`.
    pub ended_at_ms: Option<u128>,
    /// Where the simulator writes step_NNNNNN/ output directories. The UI
    /// uses this to auto-load plots for completed runs without making the
    /// user pick the directory manually.
    pub output_dir: PathBuf,
    pub remote: Option<crate::commands::remote_compute::RemoteRun>,
    pub remote_termination_unconfirmed: bool,
    pub remote_transfer_in_progress: bool,
    pub transport_exited: bool,
    pub runtime_log_evidence: Arc<Mutex<crate::commands::provenance::RuntimeLogEvidence>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    Running,
    Completed,
    Failed,
    Aborted,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunProgress {
    pub sim_time_s: f64,
    pub step: Option<u64>,
    pub vel_max: Option<f64>,
}

pub type SharedRun = Arc<Mutex<RunRecord>>;

#[derive(Default)]
pub struct AppState {
    pub runs: Mutex<HashMap<String, SharedRun>>,
    pub simulator_binary: Mutex<Option<PathBuf>>,
    /// Cached result of the GPU device probe. The probe shells out to
    /// `system_profiler` / `lspci`, which takes ~100 ms — too slow to
    /// re-run on every `simulator_status` poll, so we memoize it on
    /// first call. `Some(None)` means "we probed and found nothing"
    /// (distinct from "haven't probed yet", which is the outer None).
    pub gpu_probe: Mutex<Option<Option<String>>>,
}
