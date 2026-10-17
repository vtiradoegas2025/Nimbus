import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  clearFieldCache,
  getOrLoadField,
  setFieldCacheLimit,
} from "../../../../ui/src/lib/core/fieldCache.ts";
import {
  fieldLoadScheduler,
  LoadScheduler,
  SupersededLoadError,
} from "../../../../ui/src/lib/core/loadScheduler.ts";
import {
  getResourceTelemetrySnapshot,
  RESOURCE_TELEMETRY_SCHEMA,
  subscribeResourceTelemetry,
} from "../../../../ui/src/lib/studio/resourceTelemetry.ts";
import { visualizationResourceGovernor } from "../../../../ui/src/lib/core/visualizationResourceGovernor.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(`FAIL ${label}`);
}

function field(bytes: number) {
  return { shape: [bytes / 4], data: new Float32Array(bytes / 4) };
}

const HERE = dirname(fileURLToPath(import.meta.url));
const CAPABILITIES_SOURCE = resolve(HERE, "../../../../ui/src-tauri/src/commands/outputs.rs");

/**
 * How each `NativeFormatCapability` key in the Rust source must be named in the
 * user-facing boundary note. A format the backend declares but this map does
 * not know is a failure, not a skip: it means the backend grew a read path that
 * the panel silently says nothing about.
 */
const FORMAT_PROSE_TOKENS: Readonly<Record<string, string>> = {
  npy: "NPY",
  zfp3d: "ZFP v3",
  netcdf_classic: "classic-netCDF",
  netcdf4: "netCDF4",
};

/** The marker each declared cancellation mode must carry, and the one it must not. */
const CANCELLATION_MARKERS: Readonly<Record<string, { required: string; forbidden: string }>> = {
  cooperative: { required: "cooperativ", forbidden: "non-interruptible" },
  boundary_only: { required: "non-interruptible", forbidden: "cooperativ" },
};

function parseDeclaredCancellation(source: string): Map<string, string> {
  const declared = new Map<string, string>();
  const blocks = source.matchAll(/(\w+):\s*NativeFormatCapability\s*\{([^}]*)\}/g);
  for (const [, format, body] of blocks) {
    const mode = /cancellation:\s*"([a-z_]+)"/.exec(body);
    if (mode) declared.set(format, mode[1]);
  }
  return declared;
}

/**
 * The boundary note is hand-written prose describing a contract the backend
 * publishes through `native_field_read_capabilities`. Nothing forces the two to
 * agree, and they have already diverged once: a rewrite collapsed the netCDF
 * variants into a single "all native netCDF variants are non-interruptible"
 * claim, which is wrong — classic netCDF checks for cancellation between bounded
 * rows (`datasets.rs`, the `ClassicRows` loop) and only netCDF4's `nc_get_vars`
 * call lacks an abort hook. Understating cancellation is not harmless: it tells
 * a user a long read cannot be interrupted when it can.
 */
function checkBoundaryProseMatchesBackend(nativeIpc: string): void {
  if (!existsSync(CAPABILITIES_SOURCE)) {
    failures.push(
      `FAIL SKIPPED-AS-FAILURE backend capabilities not found at ${CAPABILITIES_SOURCE};`
      + " the boundary note could not be verified against the Rust contract",
    );
    return;
  }
  const declared = parseDeclaredCancellation(readFileSync(CAPABILITIES_SOURCE, "utf8"));

  // Guard against a regex that silently matches nothing: a vacuous parse would
  // pass every per-format assertion below without reading the contract at all.
  check(declared.size >= 4, `backend declares native format capabilities (found ${declared.size})`);

  const sentences = nativeIpc.split(/(?<=\.)\s+/);
  for (const [format, mode] of declared) {
    const token = FORMAT_PROSE_TOKENS[format];
    check(Boolean(token), `boundary note knows how to name backend format '${format}'`);
    if (!token) continue;

    const markers = CANCELLATION_MARKERS[mode];
    check(Boolean(markers), `backend cancellation mode '${mode}' for '${format}' is a known mode`);
    if (!markers) continue;

    const mentions = sentences.filter((sentence) => sentence.includes(token));
    check(mentions.length > 0, `boundary note mentions ${token}`);
    for (const sentence of mentions) {
      check(
        sentence.includes(markers.required) && !sentence.includes(markers.forbidden),
        `boundary note describes ${token} as ${mode} (backend declares ${mode})`,
      );
    }
  }
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

async function main(): Promise<void> {
  const initial = getResourceTelemetrySnapshot();
  check(initial.schema === RESOURCE_TELEMETRY_SCHEMA && initial.version === 6, "combined telemetry is schema-versioned");
  check(Object.isFrozen(initial) && Object.isFrozen(initial.parsedCache) && Object.isFrozen(initial.loader)
    && Object.isFrozen(initial.visualization) && Object.isFrozen(initial.preparedRegistry)
    && Object.isFrozen(initial.extractionPool),
  "combined and source snapshots are immutable");
  check(getResourceTelemetrySnapshot() === initial, "snapshot identity is stable between transitions");
  check(
    initial.boundary.nativeIpc.includes("classic-netCDF window/LOD reads stop cooperatively")
      && initial.boundary.nativeIpc.includes("netCDF4 report their non-interruptible native-call boundaries"),
    "native IPC cancellation modes are explicit",
  );
  checkBoundaryProseMatchesBackend(initial.boundary.nativeIpc);
  check(
    initial.boundary.visualization.includes("exact decoded-cache, prepared, worker-dataset, and retained-geometry"),
    "visualization telemetry distinguishes exact ownership from unavailable stages",
  );
  check(
    initial.boundary.visualization.includes("opaque vtk.js-native, driver, and GPU residency remains unavailable"),
    "resource telemetry does not turn exposed scratch into a total process-memory claim",
  );

  let notifications = 0;
  const unsubscribe = subscribeResourceTelemetry(() => { notifications += 1; });
  clearFieldCache();
  setFieldCacheLimit(128);
  const beforeLoad = getResourceTelemetrySnapshot();
  check(beforeLoad.sequence > initial.sequence && notifications >= 2, "cache transitions publish without polling");
  check(beforeLoad.parsedCache.limit === 128, "configured parsed-cache limit is visible");

  const pending = deferred<ReturnType<typeof field>>();
  const loading = getOrLoadField("telemetry-a", () => pending.promise);
  check(getResourceTelemetrySnapshot().parsedCache.pendingLoads === 1, "parsed-cache pending load is visible");
  pending.resolve(field(64));
  await loading;
  const resident = getResourceTelemetrySnapshot().parsedCache;
  check(resident.entries === 1 && resident.bytes === 64, "resident parsed bytes and entries are exact");
  await getOrLoadField("telemetry-a", async () => field(64));
  check(getResourceTelemetrySnapshot().parsedCache.hits >= resident.hits + 1, "cache hit counter advances");
  clearFieldCache();
  check(getResourceTelemetrySnapshot().parsedCache.bytes === 0, "clear releases resident parsed arrays");

  const scheduler = new LoadScheduler(1, 100);
  let schedulerNotifications = 0;
  scheduler.subscribe(() => { schedulerNotifications += 1; });
  const gate = deferred<number>();
  const active = scheduler.schedule({ estimatedBytes: 70 }, () => gate.promise);
  const queued = scheduler.schedule({ estimatedBytes: 40 }, async () => 2);
  const busy = scheduler.stats();
  check(busy.active === 1 && busy.activeBytes === 70, "scheduler reports active jobs and reserved bytes");
  check(busy.queued === 1 && busy.queuedBytes === 40, "scheduler reports queued jobs and estimates");
  check(scheduler.stats() === busy, "scheduler snapshot identity is stable between transitions");
  gate.resolve(1);
  await active;
  await queued;
  check(scheduler.stats().completed === 2 && scheduler.stats().activeBytes === 0, "completed work and released reservations are counted");
  check(schedulerNotifications >= 3, "scheduler subscriptions receive state transitions");

  const failureBase = scheduler.stats().failures;
  await scheduler.schedule({ estimatedBytes: 1 }, async () => { throw new Error("expected"); }).catch(() => undefined);
  check(scheduler.stats().failures === failureBase + 1, "loader failures are counted");

  const blocker = deferred<number>();
  const first = scheduler.schedule({ estimatedBytes: 100 }, () => blocker.promise);
  const cancelled = scheduler.schedule({ estimatedBytes: 1 }, async () => 3);
  const cancelBase = scheduler.stats().cancelled;
  scheduler.cancelQueued();
  check(await cancelled.then(() => false, (error) => error instanceof SupersededLoadError), "cancelled queued work rejects with scheduler cancellation error");
  check(scheduler.stats().cancelled === cancelBase + 1, "queued cancellation is counted separately");
  blocker.resolve(4);
  await first;

  const latestBlocker = deferred<number>();
  const running = scheduler.schedule({ estimatedBytes: 100 }, () => latestBlocker.promise);
  const obsolete = scheduler.schedule({ estimatedBytes: 1, latestScope: "telemetry-lane" }, async () => 5);
  const supersededBase = scheduler.stats().superseded;
  const newest = scheduler.schedule({ estimatedBytes: 1, latestScope: "telemetry-lane" }, async () => 6);
  check(await obsolete.then(() => false, (error) => error instanceof SupersededLoadError), "new latest request rejects its queued predecessor");
  check(scheduler.stats().superseded === supersededBase + 1, "superseded work has a distinct counter");
  latestBlocker.resolve(7);
  await running;
  await newest;

  // Verify the shared scheduler is wired into the combined source-neutral view.
  const globalGate = deferred<number>();
  const globalBefore = getResourceTelemetrySnapshot();
  const globalJob = fieldLoadScheduler.schedule({ estimatedBytes: 4096 }, () => globalGate.promise);
  const globalBusy = getResourceTelemetrySnapshot();
  check(globalBusy.loader.active === globalBefore.loader.active + 1, "combined telemetry observes shared active loader jobs");
  check(globalBusy.loader.activeBytes === globalBefore.loader.activeBytes + 4096, "combined telemetry observes shared byte reservations");
  check(
    globalBusy.visualization.byStage["native-read"].reservedBytes
      === globalBefore.visualization.byStage["native-read"].reservedBytes + 4096,
    "global field loading reserves native-read capacity in the visualization ledger",
  );
  globalGate.resolve(8);
  await globalJob;
  check(getResourceTelemetrySnapshot().loader.completed === globalBefore.loader.completed + 1, "combined telemetry observes shared completions");
  check(
    getResourceTelemetrySnapshot().visualization.byStage["native-read"].reservedBytes
      === globalBefore.visualization.byStage["native-read"].reservedBytes,
    "completed field loading releases native-read capacity",
  );

  const visualizationBefore = getResourceTelemetrySnapshot().visualization;
  const visualizationReservation = visualizationResourceGovernor.reserve({
    owner: "telemetry-test",
    generation: "frame-1",
    priority: "visible",
    claims: [{ allocationId: "telemetry-prepared", stage: "prepared-main", bytes: 8192 }],
  });
  check(visualizationReservation.ok, "visualization ownership is admitted into the shared ledger");
  if (!visualizationReservation.ok) throw new Error(visualizationReservation.reason);
  check(
    getResourceTelemetrySnapshot().visualization.residentBytes
      === visualizationBefore.residentBytes + 8192,
    "combined telemetry observes exact visualization residency",
  );
  visualizationResourceGovernor.release(visualizationReservation.reservation.id);
  check(
    getResourceTelemetrySnapshot().visualization.residentBytes === visualizationBefore.residentBytes,
    "combined telemetry observes visualization release",
  );

  unsubscribe();
  const stoppedAt = notifications;
  clearFieldCache();
  check(notifications === stoppedAt, "unsubscribe stops combined notifications");
  setFieldCacheLimit(512 * 1024 * 1024);

  console.log(`\nresource telemetry tests: ${passed} passed, ${failures.length} failed.`);
  if (failures.length) {
    console.log("\nFailures:");
    for (const failure of failures) console.log(`  ${failure}`);
    process.exit(1);
  }
}

void main();
