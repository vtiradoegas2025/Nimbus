import assert from "node:assert/strict";
import type { LoadingWorkerProfile } from "../../../../ui/src/lib/studio/workerProfile.ts";

const MIB = 1024 ** 2;
const hardware = { cpuBrand: "Synthetic CPU", coreCount: 4, ramBytes: 16 * 1024 ** 3 };
const storage = new Map<string, string>();
const events = new EventTarget();
let limit = 512;
const configurations: number[] = [];
const io = (mib = limit) => ({ limitBytes: mib * MIB, reservedBytes: 0, waitingReads: 0,
  automaticBytes: 512 * MIB, maximumBytes: 4096 * MIB });
let statusReply!: (value: unknown) => void;
let configurationReply: ((value: unknown) => void) | null = null;
let holdConfiguration = false;
let finishProfile!: (value: unknown) => void;
let requestId: string | null = null;
let registered = false;
let cancellations = 0;
let profileCalls = 0;
let failWrites = false;
const bridge = async (command: string, args: Record<string, unknown>) => {
  if (command === "configure_native_io_budget") {
    const next = Number(args.limitMib || 512);
    if (next > 4096) throw new Error("Limit exceeds this computer's maximum");
    configurations.push(next);
    if (holdConfiguration) {
      holdConfiguration = false;
      await new Promise(resolve => { configurationReply = resolve; });
    }
    limit = next;
    return io();
  }
  if (command === "native_io_budget_status") return new Promise(resolve => { statusReply = resolve; });
  if (command === "profile_loading_workers") {
    profileCalls++;
    requestId = String(args.requestId);
    registered = false;
    return new Promise(resolve => { finishProfile = resolve; });
  }
  if (command === "cancel_loading_worker_profile") { cancellations++; return registered; }
  if (command === "loading_worker_profile_status") return registered ? { requestId, workers: 1, completed: 0, total: 3 } : null;
  throw new Error(`Unexpected native command: ${command}`);
};
(globalThis as Record<string, unknown>).window = {
  __TAURI_INTERNALS__: { invoke: bridge },
  addEventListener: events.addEventListener.bind(events), dispatchEvent: events.dispatchEvent.bind(events),
  localStorage: { getItem: (key: string) => storage.get(key) ?? null,
    setItem: (key: string, value: string) => {
      if (failWrites) throw new Error("Storage unavailable");
      storage.set(key, value);
    } },
};
(globalThis as Record<string, unknown>).document = { visibilityState: "visible" };
const perf = await import("../../../../ui/src/lib/studio/studioPerformance.ts");
const settings = await import("../../../../ui/src/lib/studio/studioSettings.ts");
const { fieldLoadScheduler } = await import("../../../../ui/src/lib/core/loadScheduler.ts");
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
async function waitFor(predicate: () => boolean, timeout = 3000): Promise<void> {
  const deadline = Date.now() + timeout;
  while (!predicate() && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 10));
  assert.ok(predicate(), "condition did not complete before deadline");
}
await perf.initializeStudioPerformance(hardware, true);
assert.equal(profileCalls, 0, "initialization alone must not launch profiling in qualification entry points");
perf.startAutomaticWorkerMeasurement();
await waitFor(() => profileCalls === 1);
assert.equal(perf.studioPerformanceSnapshot().phase, "running");
await perf.cancelLoadingWorkerMeasurement();
assert.equal(cancellations, 1);
registered = true;
await waitFor(() => cancellations >= 2);
assert.equal(perf.studioPerformanceSnapshot().phase, "cancelling", "early cancellation waits for native completion");
finishProfile(null);
await waitFor(() => perf.studioPerformanceSnapshot().phase === "idle");
assert.equal(perf.studioPerformanceSnapshot().profile, null, "cancelled measurements cannot be saved");

const oldBudget = io();
const refresh = perf.refreshNativeIoBudget();
await tick();
await perf.applyStudioPerformanceSettings({ ...settings.STUDIO_SETTINGS_DEFAULTS, ioBudgetMiB: 1024 });
statusReply(oldBudget);
await refresh;
assert.equal(perf.studioPerformanceSnapshot().io!.limitBytes, 1024 * MIB, "stale status must not overwrite configured limits");
assert.equal(perf.studioPerformanceSnapshot().budget!.fieldLoadBytes, 1024 * MIB);

const beforeSaves = configurations.length;
holdConfiguration = true;
const first = { ...settings.STUDIO_SETTINGS_DEFAULTS, ioBudgetMiB: 64, autoProfileWorkers: false };
const second = { ...first, ioBudgetMiB: 2048, loadingWorkers: 3 };
const saveA = perf.saveStudioPerformanceSettings(first);
await waitFor(() => configurationReply !== null);
const saveB = perf.saveStudioPerformanceSettings(second);
configurationReply!(null);
assert.deepEqual(await Promise.all([saveA, saveB]), [true, true]);
await tick();
assert.deepEqual(configurations.slice(beforeSaves), [64, 2048], "saving emits no stale preference echo");
assert.equal(settings.loadStudioSettings().ioBudgetMiB, 2048);
assert.equal(perf.studioPerformanceSnapshot().budget!.maxConcurrentLoads, 3);
for (const invalid of [{ ...second, ioBudgetMiB: 2 }, { ...second, ioBudgetMiB: 8192 },
  { ...second, loadingWorkers: 1.5 }, { ...second, loadingWorkers: 65 }]) {
  await assert.rejects(perf.saveStudioPerformanceSettings(invalid));
  assert.equal(settings.loadStudioSettings().ioBudgetMiB, 2048, "failed changes do not persist");
  assert.equal(perf.studioPerformanceSnapshot().io!.limitBytes, 2048 * MIB);
}

let releaseRead!: () => void;
const activeRead = fieldLoadScheduler.schedule({ estimatedBytes: 1024 }, () => new Promise<void>(resolve => { releaseRead = resolve; }));
await waitFor(() => fieldLoadScheduler.stats().active === 1);
await perf.measureLoadingWorkers();
assert.equal(profileCalls, 1, "a measurement must wait for active field work");
releaseRead(); await activeRead;
const measurement = perf.measureLoadingWorkers();
await waitFor(() => profileCalls === 2);
const report: LoadingWorkerProfile = { schema: "nimbus.studio.worker-profile", schemaVersion: 1,
  benchmark: "npy-load-scan-v1", hardware, recommendedWorkers: 2, bytesPerTrial: 32 * MIB, elapsedMs: 200,
  trials: [{ workers: 1, medianMs: 20, throughputMibS: 1600 }, { workers: 2, medianMs: 10, throughputMibS: 3200 }] };
finishProfile(report); await measurement;
assert.deepEqual(perf.studioPerformanceSnapshot().profile, report);
assert.equal(perf.studioPerformanceSnapshot().budget!.maxConcurrentLoads, 3, "profiling preserves a manual worker preference");
await perf.saveStudioPerformanceSettings({ ...second, loadingWorkers: 0 });
assert.equal(perf.studioPerformanceSnapshot().budget!.maxConcurrentLoads, 2, "automatic uses the measured recommendation");
failWrites = true;
assert.equal(await perf.saveStudioPerformanceSettings({ ...second, ioBudgetMiB: 512 }), false);
assert.equal(settings.loadStudioSettings().ioBudgetMiB, 2048);
assert.equal(perf.studioPerformanceSnapshot().io!.limitBytes, 512 * MIB, "session limits remain observable after storage failure");
console.log("Studio performance lifecycle, cancellation, configuration races and persistence: passed");
