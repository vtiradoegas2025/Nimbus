use super::hardware::{hardware_info, HardwareInfo};
use crate::error::{AppError, AppResult};
use once_cell::sync::Lazy;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const CELLS: usize = 64 * 128 * 64;
const JOBS: usize = 16;
const MAX_SECONDS: u64 = 20;
pub const BENCHMARK_ID: &str = "npy-load-scan-v1";
static ACTIVE: Lazy<Mutex<Option<Arc<Control>>>> = Lazy::new(|| Mutex::new(None));

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerProfileStatus {
    request_id: String,
    workers: usize,
    completed: usize,
    total: usize,
}
struct Control {
    cancelled: AtomicBool,
    status: Mutex<WorkerProfileStatus>,
}
impl Control {
    fn check(&self, start: Instant) -> AppResult<()> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(AppError::Task("Worker measurement cancelled".into()));
        }
        if start.elapsed() > Duration::from_secs(MAX_SECONDS) {
            return Err(AppError::Task(
                "Worker measurement exceeded its time limit. Retry while this computer is idle."
                    .into(),
            ));
        }
        Ok(())
    }
}
struct ActiveLease(Arc<Control>);
impl Drop for ActiveLease {
    fn drop(&mut self) {
        if let Ok(mut active) = ACTIVE.lock() {
            if active
                .as_ref()
                .map_or(false, |current| Arc::ptr_eq(current, &self.0))
            {
                *active = None;
            }
        }
    }
}
struct CancelOnDrop(Arc<Control>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancelled.store(true, Ordering::Release);
    }
}
struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
impl Fixture {
    fn create() -> AppResult<Self> {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| AppError::Task(error.to_string()))?
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "nimbus-worker-profile-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&root)?;
        let fixture = Self(root);
        let mut header =
            "{'descr': '<f4', 'fortran_order': False, 'shape': (64, 128, 64), }".to_string();
        while (10 + header.len() + 1) % 16 != 0 {
            header.push(' ');
        }
        header.push('\n');
        let mut bytes = Vec::with_capacity(10 + header.len() + CELLS * 4);
        bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for index in 0..CELLS {
            bytes.extend_from_slice(&((index % 1024) as f32 / 1024.0).to_le_bytes());
        }
        fs::write(fixture.path(), bytes)?;
        Ok(fixture)
    }
    fn path(&self) -> PathBuf {
        self.0.join("field.npy")
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerTrial {
    workers: usize,
    median_ms: f64,
    throughput_mib_s: f64,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerProfile {
    schema: &'static str,
    schema_version: u32,
    benchmark: &'static str,
    hardware: HardwareInfo,
    recommended_workers: usize,
    trials: Vec<WorkerTrial>,
    bytes_per_trial: usize,
    elapsed_ms: f64,
}

fn candidates(cores: usize) -> Vec<usize> {
    let upper = cores.clamp(1, 16);
    let mut values: Vec<_> = [1, 2, 4, 8, 16]
        .into_iter()
        .filter(|value| *value <= upper)
        .collect();
    if values.last() != Some(&upper) {
        values.push(upper);
    }
    values
}
fn recommendation(trials: &[WorkerTrial]) -> AppResult<usize> {
    let best = trials
        .iter()
        .map(|trial| trial.median_ms)
        .fold(f64::INFINITY, f64::min);
    if !best.is_finite() || best <= 0.0 {
        return Err(AppError::Task(
            "No valid worker measurements completed".into(),
        ));
    }
    trials
        .iter()
        .filter(|trial| trial.median_ms <= best / 0.95)
        .map(|trial| trial.workers)
        .min()
        .ok_or_else(|| AppError::Task("No worker recommendation could be selected".into()))
}
fn trial(
    path: &Path,
    workers: usize,
    jobs: usize,
    control: &Control,
    start: Instant,
) -> AppResult<f64> {
    control.check(start)?;
    let next = AtomicUsize::new(0);
    let measured = Instant::now();
    std::thread::scope(|scope| -> AppResult<()> {
        let mut handles = Vec::new();
        for _ in 0..workers {
            let next = &next;
            handles.push(
                std::thread::Builder::new()
                    .spawn_scoped(scope, move || -> AppResult<()> {
                        loop {
                            control.check(start)?;
                            if next.fetch_add(1, Ordering::Relaxed) >= jobs {
                                return Ok(());
                            }
                            let bytes = crate::npy::read_f32_file(path)?;
                            let layout = crate::npy::parse_f32_layout(&bytes)?;
                            let mut checksum = 0.0_f64;
                            for block in bytes[layout.data_offset..].chunks(64 * 1024) {
                                control.check(start)?;
                                for cell in block.chunks_exact(4) {
                                    checksum +=
                                        f32::from_le_bytes([cell[0], cell[1], cell[2], cell[3]])
                                            as f64;
                                }
                            }
                            if checksum != CELLS as f64 * 1023.0 / 2048.0 {
                                return Err(AppError::Task(
                                    "Worker measurement data validation failed".into(),
                                ));
                            }
                            std::hint::black_box(checksum);
                        }
                    })
                    .map_err(|error| {
                        AppError::Task(format!("Cannot create measurement worker: {error}"))
                    })?,
            );
        }
        for handle in handles {
            handle
                .join()
                .map_err(|_| AppError::Task("Measurement worker failed".into()))??;
        }
        Ok(())
    })?;
    Ok(measured.elapsed().as_secs_f64() * 1000.0)
}
fn measure(control: &Control) -> AppResult<WorkerProfile> {
    let hardware = hardware_info()?;
    let cores = hardware
        .core_count
        .map(|value| value as usize)
        .or_else(|| {
            std::thread::available_parallelism()
                .ok()
                .map(|value| value.get())
        })
        .unwrap_or(1);
    let choices = candidates(cores);
    control
        .status
        .lock()
        .map_err(|_| AppError::Task("Measurement status lock poisoned".into()))?
        .total = choices.len();
    let start = Instant::now();
    let fixture = Fixture::create()?;
    let mut trials = Vec::new();
    for workers in choices {
        control
            .status
            .lock()
            .map_err(|_| AppError::Task("Measurement status lock poisoned".into()))?
            .workers = workers;
        trial(&fixture.path(), workers, JOBS, control, start)?;
        let mut samples = Vec::new();
        for _ in 0..3 {
            samples.push(trial(&fixture.path(), workers, JOBS, control, start)?);
        }
        samples.sort_by(f64::total_cmp);
        let median_ms = samples[1];
        trials.push(WorkerTrial {
            workers,
            median_ms,
            throughput_mib_s: (JOBS * CELLS * 4) as f64 / (1024.0 * 1024.0) / (median_ms / 1000.0),
        });
        control
            .status
            .lock()
            .map_err(|_| AppError::Task("Measurement status lock poisoned".into()))?
            .completed += 1;
    }
    control.check(start)?;
    let recommended_workers = recommendation(&trials)?;
    Ok(WorkerProfile {
        schema: "nimbus.studio.worker-profile",
        schema_version: 1,
        benchmark: BENCHMARK_ID,
        hardware,
        recommended_workers,
        trials,
        bytes_per_trial: JOBS * CELLS * 4,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
    })
}

#[tauri::command]
pub async fn profile_loading_workers(request_id: String) -> AppResult<WorkerProfile> {
    if request_id.is_empty()
        || request_id.len() > 128
        || !request_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(AppError::Invalid(
            "Invalid worker measurement request ID".into(),
        ));
    }
    let control = Arc::new(Control {
        cancelled: AtomicBool::new(false),
        status: Mutex::new(WorkerProfileStatus {
            request_id,
            workers: 0,
            completed: 0,
            total: 0,
        }),
    });
    {
        let mut active = ACTIVE
            .lock()
            .map_err(|_| AppError::Task("Measurement lock poisoned".into()))?;
        if active.is_some() {
            return Err(AppError::Task(
                "A worker measurement is already running".into(),
            ));
        }
        *active = Some(Arc::clone(&control));
    }
    let _cancel_on_drop = CancelOnDrop(Arc::clone(&control));
    let lease = ActiveLease(control);
    tauri::async_runtime::spawn_blocking(move || {
        let lease = lease;
        measure(&lease.0)
    })
    .await
    .map_err(|error| AppError::Task(error.to_string()))?
}
#[tauri::command]
pub fn loading_worker_profile_status() -> AppResult<Option<WorkerProfileStatus>> {
    let active = ACTIVE
        .lock()
        .map_err(|_| AppError::Task("Measurement lock poisoned".into()))?;
    active
        .as_ref()
        .map(|control| {
            control
                .status
                .lock()
                .map(|value| value.clone())
                .map_err(|_| AppError::Task("Measurement status lock poisoned".into()))
        })
        .transpose()
}
#[tauri::command]
pub fn cancel_loading_worker_profile(request_id: String) -> AppResult<bool> {
    let active = ACTIVE
        .lock()
        .map_err(|_| AppError::Task("Measurement lock poisoned".into()))?;
    if let Some(control) = active.as_ref() {
        let status = control
            .status
            .lock()
            .map_err(|_| AppError::Task("Measurement status lock poisoned".into()))?;
        if status.request_id == request_id {
            control.cancelled.store(true, Ordering::Release);
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn public_measurement_completes_and_releases_single_flight_owner() {
        assert!(
            tauri::async_runtime::block_on(profile_loading_workers("invalid id".into())).is_err()
        );
        let profile =
            tauri::async_runtime::block_on(profile_loading_workers("bounded-test".into())).unwrap();
        assert_eq!(profile.benchmark, BENCHMARK_ID);
        assert_eq!(profile.bytes_per_trial, JOBS * CELLS * 4);
        assert_eq!(
            profile.recommended_workers,
            recommendation(&profile.trials).unwrap()
        );
        assert!(profile
            .trials
            .iter()
            .all(|trial| trial.median_ms > 0.0 && trial.throughput_mib_s > 0.0));
        assert!(loading_worker_profile_status().unwrap().is_none());
        assert!(!cancel_loading_worker_profile("bounded-test".into()).unwrap());
    }
    #[test]
    fn candidate_counts_are_bounded_and_include_the_actual_core_limit() {
        assert_eq!(candidates(0), vec![1]);
        assert_eq!(candidates(6), vec![1, 2, 4, 6]);
        assert_eq!(candidates(128), vec![1, 2, 4, 8, 16]);
    }
    #[test]
    fn recommendation_prefers_fewer_workers_near_the_best_median() {
        let trials = vec![
            WorkerTrial {
                workers: 1,
                median_ms: 100.0,
                throughput_mib_s: 1.0,
            },
            WorkerTrial {
                workers: 2,
                median_ms: 52.0,
                throughput_mib_s: 1.0,
            },
            WorkerTrial {
                workers: 4,
                median_ms: 50.0,
                throughput_mib_s: 1.0,
            },
            WorkerTrial {
                workers: 8,
                median_ms: 90.0,
                throughput_mib_s: 1.0,
            },
        ];
        assert_eq!(recommendation(&trials).unwrap(), 2);
        assert!(recommendation(&[]).is_err());
    }
    #[test]
    fn generated_read_work_is_validated_and_cancellable() {
        let fixture = Fixture::create().unwrap();
        let control = Control {
            cancelled: AtomicBool::new(false),
            status: Mutex::new(WorkerProfileStatus {
                request_id: "test".into(),
                workers: 0,
                completed: 0,
                total: 0,
            }),
        };
        assert!(trial(&fixture.path(), 2, 2, &control, Instant::now()).unwrap() > 0.0);
        control.cancelled.store(true, Ordering::Release);
        assert!(trial(&fixture.path(), 2, 2, &control, Instant::now()).is_err());
        let directory = fixture.0.clone();
        drop(fixture);
        assert!(!directory.exists());
    }
}
