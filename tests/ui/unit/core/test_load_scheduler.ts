/* Deterministic byte-bounded/latest-wins scheduler tests. */

import {
  estimateBoundedFieldWindowBytes,
  LoadScheduler,
  SupersededLoadError,
} from "../../../../ui/src/lib/core/loadScheduler.ts";
import { visualizationResourceGovernor, visualizationReservationClaim } from "../../../../ui/src/lib/core/visualizationResourceGovernor.ts";
import { visualizationStagingQueue } from "../../../../ui/src/lib/core/visualizationStagingQueue.ts";

let passed = 0;
let failed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed++;
  else { failed++; failures.push(`FAIL ${label}`); }
}
const deferred = <T>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => { resolve = r; });
  return { promise, resolve };
};
const tick = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

async function main(): Promise<void> {
  // Bounded large-domain windows reserve their admitted tile, not the full
  // source field. Three 16 MiB uniform component tiles fit concurrently while
  // terrain's explicit 256 MiB working budget remains single-file conservative.
  {
    const MIB = 1024 * 1024;
    const uniform = estimateBoundedFieldWindowBytes({
      extent: [256, 256, 64],
      stride: [1, 1, 1],
      maxOutputBytes: 16 * MIB,
      maxSourceBytes: 16 * MIB,
    }, false);
    check(uniform === 128 * MIB, "uniform window includes bounded aggregation and IPC working copies");
    check(uniform * 3 === 384 * MIB, "three vector-component tiles fit the global byte ceiling");
    const terrain = estimateBoundedFieldWindowBytes({
      extent: [64, 64, 32],
      stride: [2, 2, 2],
      maxOutputBytes: 64 * MIB,
      maxSourceBytes: 256 * MIB,
    }, true);
    check(terrain > 256 * MIB, "terrain reservation retains its explicit geometry working budget");
    const loose = estimateBoundedFieldWindowBytes({ extent: [32, 32, 16], stride: [2, 2, 2],
      maxSourceBytes: 256 * MIB, maxOutputBytes: 64 * MIB }, false);
    check(loose >= 384 * MIB, "small returned windows still reserve the budgets admitted by native code");
    check(estimateBoundedFieldWindowBytes({ extent: [4, 4, 4], stride: [1, 1, 1] }, false) >= 128 * MIB,
      "omitted request budgets match native defaults");
  }

  // Byte reservations, rather than task count alone, gate starts.
  {
    const scheduler = new LoadScheduler(1, 100);
    const controller = new AbortController();
    let rejectNative!: (error: Error) => void;
    const worker = new Promise<number>((_, reject) => { rejectNative = reject; });
    const pending = scheduler.schedule({ estimatedBytes: 80, signal: controller.signal }, () => worker).catch(error => error);
    await tick();
    controller.abort();
    check(scheduler.stats().activeBytes === 80, "native cancellation retains admission until the worker acknowledges it");
    rejectNative(new Error("background task failed: field read cancelled"));
    check((await pending).name === "AbortError", "native rejection after cancellation retains abort semantics");
    check(scheduler.stats().cancelled === 1 && scheduler.stats().failures === 0 && scheduler.stats().activeBytes === 0,
      "cancelled native work releases its bytes without reporting a plot failure");
  }

  {
    const scheduler = new LoadScheduler(4, 100);
    const first = deferred<number>();
    let secondStarted = false;
    const p1 = scheduler.schedule({ estimatedBytes: 70 }, () => first.promise);
    const p2 = scheduler.schedule({ estimatedBytes: 40 }, async () => {
      secondStarted = true;
      return 2;
    });
    await tick();
    check(scheduler.stats().active === 1, "byte cap holds second task queued");
    check(secondStarted === false, "over-budget task has not started");
    first.resolve(1);
    check(await p1 === 1, "first task resolves");
    check(await p2 === 2 && secondStarted, "queued task starts after bytes release");
  }

  {
    const scheduler = new LoadScheduler(1, 100);
    const running = deferred<number>();
    const first = scheduler.schedule({ estimatedBytes: 80 }, () => running.promise);
    const controller = new AbortController();
    let cancelledStarted = false;
    const cancelled = scheduler.schedule({ estimatedBytes: 20, signal: controller.signal }, async () => {
      cancelledStarted = true;
      return 2;
    }).catch(error => error);
    controller.abort();
    check((await cancelled).name === "AbortError", "aborted queued work settles without waiting for running reads");
    check(scheduler.stats().queued === 0 && !cancelledStarted, "aborted queued work cannot consume a future slot");
    running.resolve(1);
    await first;
    check(scheduler.stats().cancelled === 1 && scheduler.stats().activeBytes === 0,
      "queued cancellation retires its accounting exactly once");
  }

  {
    const scheduler = new LoadScheduler(1, 100);
    const running = deferred<number>();
    const controller = new AbortController();
    const cancelled = scheduler.schedule({ estimatedBytes: 80, signal: controller.signal }, () => running.promise)
      .catch(error => error);
    controller.abort();
    check(scheduler.stats().activeBytes === 80, "active cancellation retains its memory reservation until native work settles");
    running.resolve(1);
    check((await cancelled).name === "AbortError", "a cancelled active result cannot reach the field cache");
    check(scheduler.stats().activeBytes === 0 && scheduler.stats().failures === 0,
      "cooperative cancellation releases memory without a load failure");
  }

  {
    const scheduler = new LoadScheduler(1, 100);
    const running = deferred<number>();
    const first = scheduler.schedule({ estimatedBytes: 60 }, () => running.promise);
    const second = scheduler.schedule({ estimatedBytes: 60 }, async () => 2);
    scheduler.setLimits(2, 200);
    check(await second === 2, "a larger hardware budget admits queued work without restarting reads");
    scheduler.setLimits(1, 30);
    const third = scheduler.schedule({ estimatedBytes: 20 }, async () => 3);
    check(scheduler.stats().activeBytes === 60 && scheduler.stats().queued === 1,
      "a smaller budget drains active work before admitting another read");
    running.resolve(1);
    await first;
    check(await third === 3, "queued reads resume after the lower budget is satisfied");
    check(scheduler.stats().activeBytes === 0, "budget changes retain balanced reservations");
  }

  // Newest request removes queued obsolete work for the same widget lane.
  {
    const scheduler = new LoadScheduler(1, 100);
    const blocker = deferred<number>();
    const p0 = scheduler.schedule({ estimatedBytes: 1 }, () => blocker.promise);
    const old = scheduler.schedule(
      { estimatedBytes: 1, latestScope: "slice" },
      async () => 1,
    );
    const latest = scheduler.schedule(
      { estimatedBytes: 1, latestScope: "slice" },
      async () => 2,
    );
    const oldRejected = old.then(
      () => false,
      (error) => error instanceof SupersededLoadError,
    );
    check(await oldRejected, "queued old generation rejected as superseded");
    check(scheduler.stats().queued === 1, "only latest widget request remains queued");
    blocker.resolve(0);
    await p0;
    check(await latest === 2, "latest widget request executes");
    check(scheduler.stats().latestScopes === 0, "completed widget lanes release latest-wins bookkeeping");
  }

  // A running obsolete result is rejected instead of reaching cache/React.
  {
    const scheduler = new LoadScheduler(2, 100);
    const oldTask = deferred<number>();
    const old = scheduler.schedule(
      { estimatedBytes: 10, latestScope: "volume" },
      () => oldTask.promise,
    );
    const latest = scheduler.schedule(
      { estimatedBytes: 10, latestScope: "volume" },
      async () => 9,
    );
    check(await latest === 9, "new request can execute beside running old request");
    oldTask.resolve(3);
    check(
      await old.then(() => false, (error) => error instanceof SupersededLoadError),
      "running obsolete result is discarded",
    );
  }

  // One field larger than the byte cap progresses alone.
  {
    const scheduler = new LoadScheduler(3, 10);
    const value = await scheduler.schedule({ estimatedBytes: 50 }, async () => 7);
    check(value === 7, "single oversized load makes progress");
    check(scheduler.stats().activeBytes === 0, "oversized reservation released");
  }

  // Speculation can consume spare capacity but never the final interactive slot.
  {
    const scheduler = new LoadScheduler(3, 1_000);
    const first = deferred<number>();
    const second = deferred<number>();
    const b1 = scheduler.schedule(
      { estimatedBytes: 10, priority: "background" },
      () => first.promise,
    );
    const b2 = scheduler.schedule(
      { estimatedBytes: 10, priority: "background" },
      () => second.promise,
    );
    let thirdStarted = false;
    const b3 = scheduler.schedule(
      { estimatedBytes: 10, priority: "background" },
      async () => { thirdStarted = true; return 3; },
    );
    await tick();
    check(scheduler.stats().activeBackground === 2, "background work leaves one interactive slot reserved");
    check(!thirdStarted, "third speculative read remains queued");
    const interactive = scheduler.schedule(
      { estimatedBytes: 10, priority: "interactive" },
      async () => 9,
    );
    check(await interactive === 9, "interactive read starts immediately beside speculation");
    first.resolve(1);
    second.resolve(2);
    await Promise.all([b1, b2]);
    check(await b3 === 3, "queued speculation resumes only after capacity returns");
  }

  // A cache hit can advance latest-wins identity without scheduling work.
  {
    const scheduler = new LoadScheduler(1, 100);
    const running = deferred<number>();
    const old = scheduler.schedule(
      { estimatedBytes: 10, latestScope: "cached-lane" },
      () => running.promise,
    );
    scheduler.supersede("cached-lane");
    check(scheduler.stats().queued === 0, "cached supersede schedules no work");
    running.resolve(1);
    check(
      await old.then(() => false, (error) => error instanceof SupersededLoadError),
      "cached newer frame invalidates a running predecessor",
    );
    check(scheduler.stats().latestScopes === 0, "cache-hit supersession retires after obsolete work settles");
  }

  // Beast-scale fields reserve their real decoded size without allocating the
  // payload in this deterministic test. Two 168 MiB reads fit below Nimbus's
  // 384 MiB cap; a third waits, and a newer request supersedes that queued
  // read before it can consume memory or IPC bandwidth.
  {
    const MIB = 1024 * 1024;
    const scheduler = new LoadScheduler(3, 384 * MIB);
    const first = deferred<number>();
    const second = deferred<number>();
    let obsoleteStarted = false;
    const p1 = scheduler.schedule(
      { estimatedBytes: 168 * MIB },
      () => first.promise,
    );
    const p2 = scheduler.schedule(
      { estimatedBytes: 168 * MIB },
      () => second.promise,
    );
    const obsolete = scheduler.schedule(
      { estimatedBytes: 168 * MIB, latestScope: "mounted-slice" },
      async () => {
        obsoleteStarted = true;
        return 3;
      },
    );
    await tick();
    check(scheduler.stats().active === 2, "two delayed 168 MiB reads run together");
    check(
      scheduler.stats().activeBytes === 336 * MIB,
      "168 MiB reads reserve their decoded-byte equivalent",
    );
    check(scheduler.stats().queued === 1, "third 168 MiB read waits below 384 MiB cap");

    const newest = scheduler.schedule(
      { estimatedBytes: 8 * MIB, latestScope: "mounted-slice" },
      async () => 4,
    );
    check(
      await obsolete.then(() => false, (error) => error instanceof SupersededLoadError),
      "new identity rejects queued beast-scale predecessor",
    );
    check(obsoleteStarted === false, "superseded 168 MiB read never starts");
    first.resolve(1);
    check(await p1 === 1, "first delayed beast-scale read resolves");
    check(await newest === 4, "newest identity starts when one reservation releases");
    second.resolve(2);
    check(await p2 === 2, "second delayed beast-scale read resolves");
    check(scheduler.stats().activeBytes === 0, "beast-scale reservations fully release");
  }

  {
    let pressureId = 0;
    visualizationStagingQueue.schedule({
      owner: "staging-pressure", generation: "1", priority: "visible",
      claims: [visualizationReservationClaim("staging-pressure", "worker-scratch", 128 * 1024 * 1024)],
    }, (pressure) => {
      if (!pressure.ok) throw new Error(pressure.reason);
      pressureId = pressure.reservation.id;
    });
    const scheduler = new LoadScheduler(3, 384 * 1024 * 1024, "staging-read-test");
    let started = 0;
    const older = scheduler.schedule({ estimatedBytes: 16, latestScope: "waiting-frame" }, async () => ++started);
    const olderResult = older.catch((error) => error);
    const newer = scheduler.schedule({ estimatedBytes: 16, latestScope: "waiting-frame" }, async () => ++started);
    const newerResult = newer.catch((error) => error);
    check(await olderResult instanceof SupersededLoadError, "supersession cancels a read waiting for preparation memory");
    scheduler.cancelQueued();
    check(await newerResult instanceof SupersededLoadError, "queued cancellation includes shared staging waiters");
    visualizationResourceGovernor.release(pressureId);
    await tick();
    check(started === 0, "cancelled staging waiters never start native reads");
    check(scheduler.stats().activeBytes === 0 && scheduler.stats().latestScopes === 0,
      "cancelled staging waiters retire scheduler accounting and scopes");
  }

  {
    let pressureId = 0;
    visualizationStagingQueue.schedule({
      owner: "abort-staging-pressure", generation: "1", priority: "visible",
      claims: [visualizationReservationClaim("abort-staging-pressure", "worker-scratch", 128 * 1024 * 1024)],
    }, pressure => {
      if (!pressure.ok) throw new Error(pressure.reason);
      pressureId = pressure.reservation.id;
    });
    const scheduler = new LoadScheduler(1, 100, "abort-staging-test");
    const controller = new AbortController();
    let started = false;
    const pending = scheduler.schedule({ estimatedBytes: 100, signal: controller.signal }, async () => {
      started = true;
      return 1;
    }).catch(error => error);
    controller.abort();
    check((await pending).name === "AbortError", "signal cancellation retires a read waiting on staging");
    visualizationResourceGovernor.release(pressureId);
    await tick();
    check(!started && scheduler.stats().activeBytes === 0, "cancelled staging reads never start after pressure releases");
  }

  console.log(`\nloadScheduler tests: ${passed} passed, ${failed} failed.`);
  if (failures.length) {
    console.log("\nFailures:");
    for (const failure of failures) console.log(`  ${failure}`);
    process.exit(1);
  }
}

void main();
