import type { HardwareInfo } from "@/lib/core/types";
import { fieldLoadScheduler } from "@/lib/core/loadScheduler";
import { configurePlaybackResources, type PlaybackResourceBudget } from "@/lib/core/playbackResourcePolicy";
import { invokeCommand } from "@/lib/platform/appInvoke";
import { publishNativeIoBudgetLimit } from "@/lib/core/nativeWindowBudget";
import { loadStudioSettings, saveStudioSettings, STUDIO_SETTINGS_CHANGE_EVENT, type StudioSettings } from "./studioSettings";
import { hardwareProfileIdentity, loadWorkerProfile, parseWorkerProfile, saveWorkerProfile, type LoadingWorkerProfile } from "./workerProfile";

export interface NativeIoBudgetInfo {
  limitBytes: number;
  reservedBytes: number;
  waitingReads: number;
  automaticBytes: number;
  maximumBytes: number;
}
export interface WorkerMeasurementStatus { requestId: string; workers: number; completed: number; total: number }
export interface StudioPerformanceSnapshot {
  native: boolean;
  hardware: HardwareInfo | null;
  io: NativeIoBudgetInfo | null;
  budget: PlaybackResourceBudget | null;
  profile: LoadingWorkerProfile | null;
  phase: "idle" | "running" | "cancelling";
  status: WorkerMeasurementStatus | null;
  error: string | null;
  configError: string | null;
  profileSaved: boolean;
}
let snapshot: StudioPerformanceSnapshot = { native: false, hardware: null, io: null, budget: null,
  profile: null, phase: "idle", status: null, error: null, configError: null, profileSaved: true };
const listeners = new Set<() => void>();
let configuration = Promise.resolve();
let suppressPreferenceEcho = false;
let currentSettings: StudioSettings | null = null;
let configuring = false;
let budgetGeneration = 0;
let statusSequence = 0;
let automaticAttempted = false;
let autoTimer: ReturnType<typeof setTimeout> | null = null;
let activeRequest: string | null = null;
let sequence = 0;
let installed = false;
let appStarted = false;
const MIB = 1024 ** 2;

export function studioPerformanceSnapshot(): StudioPerformanceSnapshot { return snapshot; }
export function subscribeStudioPerformance(listener: () => void): () => void {
  listeners.add(listener); return () => listeners.delete(listener);
}
function publish(values: Partial<StudioPerformanceSnapshot>): void {
  snapshot = { ...snapshot, ...values }; for (const listener of listeners) listener();
}
function ioInfo(value: NativeIoBudgetInfo): NativeIoBudgetInfo {
  if (!value || ![value.limitBytes, value.automaticBytes, value.maximumBytes].every(n => Number.isSafeInteger(n) && n > 0)
      || ![value.reservedBytes, value.waitingReads].every(n => Number.isSafeInteger(n) && n >= 0)
      || value.limitBytes > value.maximumBytes || value.automaticBytes > value.maximumBytes) throw new Error("Invalid native I/O budget response");
  return value;
}
export async function refreshNativeIoBudget(): Promise<void> {
  if (!snapshot.native || configuring) return;
  const generation = budgetGeneration;
  const sequence = ++statusSequence;
  const io = ioInfo(await invokeCommand<NativeIoBudgetInfo>("native_io_budget_status"));
  if (!configuring && generation === budgetGeneration && sequence === statusSequence) publish({ io });
}
async function applyPreferencesNow(settings: StudioSettings): Promise<void> {
  if (!Number.isSafeInteger(settings.loadingWorkers) || settings.loadingWorkers < 0 || settings.loadingWorkers > 64) throw new Error("Loading workers must be 0 (automatic) or an integer from 1 to 64.");
  if (!Number.isSafeInteger(settings.ioBudgetMiB) || settings.ioBudgetMiB !== 0
      && (settings.ioBudgetMiB < 64 || settings.ioBudgetMiB > 16384)) throw new Error("I/O memory must be 0 (automatic) or an integer from 64 to 16384 MiB.");
  const hardware = snapshot.hardware;
  if (!hardware) return;
  configuring = true;
  budgetGeneration += 1;
  try {
    const io = snapshot.native
      ? ioInfo(await invokeCommand<NativeIoBudgetInfo>("configure_native_io_budget", { limitMib: settings.ioBudgetMiB }))
      : { limitBytes: (settings.ioBudgetMiB || 512) * MIB, reservedBytes: 0, waitingReads: 0, automaticBytes: 512 * MIB, maximumBytes: 16384 * MIB };
    const budget = configurePlaybackResources({ source: snapshot.native ? "tauri" : "browser",
      systemRamBytes: hardware.ramBytes, logicalCores: hardware.coreCount }, {
      ioBytes: io.limitBytes,
      loadingWorkers: settings.loadingWorkers || snapshot.profile?.recommendedWorkers,
    });
    currentSettings = settings;
    publish({ io, budget, configError: null });
    publishNativeIoBudgetLimit(io.limitBytes);
  } finally {
    configuring = false;
    budgetGeneration += 1;
  }
}
function enqueueConfiguration<T>(operation: () => Promise<T>): Promise<T> {
  const result = configuration.then(operation);
  configuration = result.then(() => undefined, () => undefined);
  return result;
}
export function applyStudioPerformanceSettings(settings: StudioSettings): Promise<void> {
  return enqueueConfiguration(() => applyPreferencesNow(settings));
}
export function saveStudioPerformanceSettings(settings: StudioSettings): Promise<boolean> {
  return enqueueConfiguration(async () => {
    await applyPreferencesNow(settings);
    suppressPreferenceEcho = true;
    try { return saveStudioSettings(settings); }
    finally { suppressPreferenceEcho = false; }
  });
}
function reapplyCurrentSettings(): Promise<void> {
  return enqueueConfiguration(() => applyPreferencesNow(currentSettings ?? loadStudioSettings()));
}

function scheduleAutomaticMeasurement(): void {
  if (!appStarted || autoTimer != null || automaticAttempted || !snapshot.native || snapshot.profile
      || !(currentSettings ?? loadStudioSettings()).autoProfileWorkers) return;
  autoTimer = setTimeout(() => {
    autoTimer = null;
    if (!(currentSettings ?? loadStudioSettings()).autoProfileWorkers || snapshot.profile || automaticAttempted) return;
    if (document.visibilityState !== "visible" || fieldLoadScheduler.stats().active > 0
        || fieldLoadScheduler.stats().queued > 0) { scheduleAutomaticMeasurement(); return; }
    automaticAttempted = true;
    void measureLoadingWorkers();
  }, 1500);
}
export async function initializeStudioPerformance(hardware: HardwareInfo, native: boolean): Promise<void> {
  const profile = loadWorkerProfile(hardware);
  publish({ hardware, native, profile });
  try { await applyStudioPerformanceSettings(loadStudioSettings()); }
  catch (error) {
    try {
      await applyStudioPerformanceSettings({ ...loadStudioSettings(), ioBudgetMiB: 0 });
      publish({ configError: `The saved I/O limit could not be applied. Automatic limits are active. ${String(error)}` });
    } catch (fallbackError) {
      publish({ configError: `Studio could not verify the active I/O limits. ${String(fallbackError)}` });
    }
  }
  if (!installed && typeof window !== "undefined") {
    installed = true;
    window.addEventListener(STUDIO_SETTINGS_CHANGE_EVENT, () => {
      if (!suppressPreferenceEcho) void applyStudioPerformanceSettings(loadStudioSettings()).catch(error => publish({ configError: String(error) }));
      scheduleAutomaticMeasurement();
    });
    window.addEventListener("pagehide", () => { void cancelLoadingWorkerMeasurement(); });
  }
}
export function startAutomaticWorkerMeasurement(): void {
  appStarted = true;
  scheduleAutomaticMeasurement();
}
export async function measureLoadingWorkers(): Promise<void> {
  if (!snapshot.native || !snapshot.hardware || snapshot.phase !== "idle") return;
  if (fieldLoadScheduler.stats().active > 0 || fieldLoadScheduler.stats().queued > 0) {
    publish({ error: "Wait for data loading to finish before measuring workers." }); return;
  }
  automaticAttempted = true;
  const requestId = `workers-${Date.now()}-${++sequence}`;
  activeRequest = requestId;
  const hardware = snapshot.hardware;
  publish({ phase: "running", status: { requestId, workers: 0, completed: 0, total: 0 }, error: null });
  const timer = setInterval(() => {
    void invokeCommand<WorkerMeasurementStatus | null>("loading_worker_profile_status").then(status => {
      if (activeRequest === requestId && status?.requestId === requestId) {
        publish({ status });
        if (snapshot.phase === "cancelling") {
          void invokeCommand("cancel_loading_worker_profile", { requestId }).catch(() => undefined);
        }
      }
    }).catch(() => undefined);
  }, 250);
  try {
    const raw = await invokeCommand<unknown>("profile_loading_workers", { requestId });
    if (activeRequest !== requestId || studioPerformanceSnapshot().phase === "cancelling") return;
    const profile = parseWorkerProfile(raw);
    if (!profile || hardwareProfileIdentity(profile.hardware) !== hardwareProfileIdentity(hardware)) throw new Error("Worker measurement does not match this computer");
    const profileSaved = saveWorkerProfile(profile);
    publish({ profile, profileSaved });
    await reapplyCurrentSettings();
  } catch (error) {
    if (studioPerformanceSnapshot().phase !== "cancelling") publish({ error: error instanceof Error ? error.message : String(error) });
  } finally {
    clearInterval(timer);
    if (activeRequest === requestId) { activeRequest = null; publish({ phase: "idle", status: null }); }
  }
}
export async function cancelLoadingWorkerMeasurement(): Promise<void> {
  if (!activeRequest || snapshot.phase === "cancelling") return;
  const requestId = activeRequest;
  publish({ phase: "cancelling" });
  try { await invokeCommand("cancel_loading_worker_profile", { requestId }); }
  catch (error) { publish({ error: `Could not request cancellation: ${String(error)}` }); }
}
