use crate::error::{AppError, AppResult};
use once_cell::sync::Lazy;
use serde::Serialize;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

const MIB: usize = 1024 * 1024;
const DEFAULT_IO_BYTES: usize = 512 * 1024 * 1024;
static SYSTEM_RAM_BYTES: Lazy<Option<u64>> = Lazy::new(super::hardware::get_ram_bytes);
static NATIVE_IO: Lazy<Arc<NativeIoBudget>> =
    Lazy::new(|| Arc::new(NativeIoBudget::new(DEFAULT_IO_BYTES)));

struct BudgetState {
    limit: usize,
    reserved: usize,
    waiting: usize,
}
struct NativeIoBudget {
    state: Mutex<BudgetState>,
    available: Condvar,
}

pub(crate) struct NativeWindowIoReservation {
    budget: Arc<NativeIoBudget>,
    bytes: usize,
}
impl Drop for NativeWindowIoReservation {
    fn drop(&mut self) {
        if let Ok(mut state) = self.budget.state.lock() {
            debug_assert!(state.reserved >= self.bytes);
            state.reserved = state.reserved.saturating_sub(self.bytes);
            self.budget.available.notify_all();
        }
    }
}

impl NativeIoBudget {
    fn new(limit: usize) -> Self {
        Self {
            state: Mutex::new(BudgetState {
                limit,
                reserved: 0,
                waiting: 0,
            }),
            available: Condvar::new(),
        }
    }
    fn set_limit(&self, limit: usize) -> AppResult<()> {
        if limit == 0 {
            return Err(AppError::Invalid(
                "native I/O budget must be positive".into(),
            ));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| AppError::Task("native I/O budget lock poisoned".into()))?;
        // Existing workers retain ownership when a preference lowers the cap.
        state.limit = limit;
        self.available.notify_all();
        Ok(())
    }
    fn reserve(
        self: &Arc<Self>,
        bytes: usize,
        check_cancelled: impl Fn() -> AppResult<()>,
    ) -> AppResult<NativeWindowIoReservation> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| AppError::Task("native I/O budget lock poisoned".into()))?;
        let mut waiting = false;
        let result = loop {
            if let Err(error) = check_cancelled() {
                break Err(error);
            }
            if bytes == 0 || bytes > state.limit {
                break Err(AppError::Invalid(format!("native window I/O request needs {bytes} bytes; limit is {}. Increase the I/O memory limit in Settings > Performance or select a smaller region.", state.limit)));
            }
            if state
                .reserved
                .checked_add(bytes)
                .map_or(false, |next| next <= state.limit)
            {
                state.reserved += bytes;
                break Ok(NativeWindowIoReservation {
                    budget: Arc::clone(self),
                    bytes,
                });
            }
            if !waiting {
                state.waiting += 1;
                waiting = true;
            }
            state = self
                .available
                .wait_timeout(state, Duration::from_millis(25))
                .map_err(|_| AppError::Task("native I/O budget lock poisoned".into()))?
                .0;
        };
        if waiting {
            state.waiting -= 1;
        }
        result
    }
}

pub(crate) fn reservation_bytes(source_bytes: usize, output_bytes: usize) -> AppResult<usize> {
    source_bytes
        .checked_add(output_bytes.checked_mul(2).ok_or_else(|| {
            AppError::Invalid("native window I/O reservation size overflow".into())
        })?)
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| {
            AppError::Invalid("native window I/O reservation size is zero or overflows".into())
        })
}
pub(crate) fn reserve_native_window_io(
    source_bytes: usize,
    output_bytes: usize,
    check_cancelled: impl Fn() -> AppResult<()>,
) -> AppResult<NativeWindowIoReservation> {
    NATIVE_IO.reserve(
        reservation_bytes(source_bytes, output_bytes)?,
        check_cancelled,
    )
}
pub(crate) fn wake_waiters() {
    NATIVE_IO.available.notify_all();
}

pub(crate) struct NativeWindowReadResult {
    bytes: Vec<u8>,
    reservation: NativeWindowIoReservation,
}

impl NativeWindowReadResult {
    pub(crate) fn into_response(self) -> tauri::ipc::Response {
        let response = tauri::ipc::Response::new(self.bytes);
        drop(self.reservation);
        response
    }
}

pub(crate) fn read_with_budget(
    source_bytes: usize,
    output_bytes: usize,
    check_cancelled: impl Fn() -> AppResult<()>,
    read: impl FnOnce() -> AppResult<Vec<u8>>,
) -> AppResult<NativeWindowReadResult> {
    let reservation = reserve_native_window_io(source_bytes, output_bytes, check_cancelled)?;
    let bytes = read()?;
    Ok(NativeWindowReadResult { bytes, reservation })
}

fn policy_limits(ram_bytes: Option<u64>) -> (usize, usize) {
    let ram = ram_bytes
        .and_then(|bytes| usize::try_from(bytes).ok())
        .filter(|bytes| *bytes > 0);
    let maximum = ram.map_or(2 * 1024 * MIB, |bytes| {
        (bytes / 4).clamp(64 * MIB, 16 * 1024 * MIB)
    });
    let automatic = ram.map_or(DEFAULT_IO_BYTES, |bytes| {
        (bytes / 32).clamp(DEFAULT_IO_BYTES, 4 * 1024 * MIB)
    });
    (automatic.min(maximum), maximum)
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeIoBudgetInfo {
    limit_bytes: usize,
    reserved_bytes: usize,
    waiting_reads: usize,
    automatic_bytes: usize,
    maximum_bytes: usize,
}
fn budget_info(automatic_bytes: usize, maximum_bytes: usize) -> AppResult<NativeIoBudgetInfo> {
    let state = NATIVE_IO
        .state
        .lock()
        .map_err(|_| AppError::Task("native I/O budget lock poisoned".into()))?;
    Ok(NativeIoBudgetInfo {
        limit_bytes: state.limit,
        reserved_bytes: state.reserved,
        waiting_reads: state.waiting,
        automatic_bytes,
        maximum_bytes,
    })
}
#[tauri::command]
pub fn configure_native_io_budget(limit_mib: Option<u64>) -> AppResult<NativeIoBudgetInfo> {
    let (automatic, maximum) = policy_limits(*SYSTEM_RAM_BYTES);
    let requested = match limit_mib {
        None | Some(0) => automatic,
        Some(value) => usize::try_from(value)
            .ok()
            .and_then(|value| value.checked_mul(MIB))
            .filter(|value| (64 * MIB..=maximum).contains(value))
            .ok_or_else(|| {
                AppError::Invalid(format!(
                    "I/O memory limit must be 64..={} MiB, or 0 for automatic",
                    maximum / MIB
                ))
            })?,
    };
    NATIVE_IO.set_limit(requested)?;
    budget_info(automatic, maximum)
}
#[tauri::command]
pub fn native_io_budget_status() -> AppResult<NativeIoBudgetInfo> {
    let (automatic, maximum) = policy_limits(*SYSTEM_RAM_BYTES);
    budget_info(automatic, maximum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    fn wait_until_queued(budget: &NativeIoBudget) {
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while budget.state.lock().unwrap().waiting == 0 {
            assert!(std::time::Instant::now() < deadline, "worker never queued");
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    #[test]
    fn contention_waits_then_reuses_released_capacity() {
        let budget = Arc::new(NativeIoBudget::new(10));
        let first = budget.reserve(8, || Ok(())).unwrap();
        let worker_budget = Arc::clone(&budget);
        let (sent, received) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            sent.send(worker_budget.reserve(4, || Ok(())).unwrap())
                .unwrap()
        });
        wait_until_queued(&budget);
        assert!(received.try_recv().is_err());
        assert_eq!(budget.state.lock().unwrap().reserved, 8);
        drop(first);
        let second = received.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.join().unwrap();
        assert_eq!(budget.state.lock().unwrap().reserved, 4);
        drop(second);
        assert_eq!(budget.state.lock().unwrap().reserved, 0);
        assert_eq!(budget.state.lock().unwrap().waiting, 0);
    }
    #[test]
    fn cancelling_waiter_does_not_release_active_worker() {
        let budget = Arc::new(NativeIoBudget::new(8));
        let first = budget.reserve(8, || Ok(())).unwrap();
        let cancelled = Arc::new(AtomicBool::new(false));
        let worker_budget = Arc::clone(&budget);
        let flag = Arc::clone(&cancelled);
        let worker = std::thread::spawn(move || {
            worker_budget
                .reserve(4, || {
                    if flag.load(Ordering::Acquire) {
                        Err(AppError::Task("cancelled".into()))
                    } else {
                        Ok(())
                    }
                })
                .is_err()
        });
        wait_until_queued(&budget);
        cancelled.store(true, Ordering::Release);
        budget.available.notify_all();
        assert!(worker.join().unwrap());
        assert_eq!(budget.state.lock().unwrap().reserved, 8);
        assert_eq!(budget.state.lock().unwrap().waiting, 0);
        drop(first);
        assert_eq!(budget.state.lock().unwrap().reserved, 0);
    }
    #[test]
    fn lowering_limit_preserves_in_flight_ownership() {
        let budget = Arc::new(NativeIoBudget::new(16));
        let first = budget.reserve(12, || Ok(())).unwrap();
        budget.set_limit(8).unwrap();
        assert_eq!(budget.state.lock().unwrap().reserved, 12);
        assert!(budget.reserve(9, || Ok(())).is_err());
        drop(first);
        assert!(budget.reserve(8, || Ok(())).is_ok());
        assert!(budget.set_limit(0).is_err());
        assert_eq!(budget.state.lock().unwrap().limit, 8);
    }
    #[test]
    fn accounting_and_hardware_limits_are_bounded() {
        assert_eq!(reservation_bytes(8, 4).unwrap(), 16);
        assert!(reservation_bytes(usize::MAX, 1).is_err());
        assert!(reservation_bytes(0, 0).is_err());
        assert_eq!(
            policy_limits(Some(8 * 1024 * 1024 * 1024)),
            (512 * MIB, 2 * 1024 * MIB)
        );
        assert_eq!(policy_limits(None), (DEFAULT_IO_BYTES, 2 * 1024 * MIB));
    }
    #[test]
    fn completed_payload_owns_capacity_until_delivery_or_discard() {
        let budget = Arc::new(NativeIoBudget::new(8));
        let payload = NativeWindowReadResult {
            bytes: vec![1, 2, 3, 4],
            reservation: budget.reserve(8, || Ok(())).unwrap(),
        };
        assert_eq!(budget.state.lock().unwrap().reserved, 8);
        let _response = payload.into_response();
        assert_eq!(budget.state.lock().unwrap().reserved, 0);
        let discarded = NativeWindowReadResult {
            bytes: vec![1, 2, 3, 4],
            reservation: budget.reserve(8, || Ok(())).unwrap(),
        };
        drop(discarded);
        assert_eq!(budget.state.lock().unwrap().reserved, 0);
    }
}
