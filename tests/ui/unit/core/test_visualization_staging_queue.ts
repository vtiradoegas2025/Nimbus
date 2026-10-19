import assert from "node:assert/strict";
import { VisualizationStagingQueue } from "../../../../ui/src/lib/core/visualizationStagingQueue.ts";
import { VisualizationResourceGovernor, visualizationReservationClaim } from "../../../../ui/src/lib/core/visualizationResourceGovernor.ts";

const governor = new VisualizationResourceGovernor();
const queue = new VisualizationStagingQueue(governor, 128);
const reservations = new Map<string, number>();
const schedule = (name: string, bytes: number) => queue.schedule({
  owner: name, generation: "1", priority: "visible",
  claims: [visualizationReservationClaim(name, "handoff", bytes)],
}, (outcome) => {
  assert.equal(outcome.ok, true);
  if (outcome.ok) reservations.set(name, outcome.reservation.id);
});
const flush = () => new Promise<void>((resolve) => queueMicrotask(resolve));
schedule("preparation", 80);
schedule("read-a", 60);
schedule("read-b", 60);
assert.deepEqual([...reservations.keys()], ["preparation"]);
assert.equal(governor.stats().reservedBytes, 80);
governor.transition(reservations.get("preparation")!, [{ allocationId: "prepared", stage: "worker-resident", bytes: 40 }]);
await flush();
assert.deepEqual([...reservations.keys()], ["preparation", "read-a", "read-b"]);
assert.equal(governor.stats().reservedBytes, 120);
assert.equal(governor.stats().residentBytes, 40);

const cancel = schedule("cancelled", 100);
schedule("oversized", 256);
schedule("after-oversized", 1);
cancel();
governor.release(reservations.get("read-a")!);
await flush();
assert.equal(reservations.has("oversized"), false);
governor.release(reservations.get("read-b")!);
await flush();
assert.equal(reservations.has("cancelled"), false);
assert.equal(reservations.has("oversized"), true);
assert.equal(reservations.has("after-oversized"), false);
governor.release(reservations.get("oversized")!);
await flush();
assert.equal(reservations.has("after-oversized"), true);
governor.release(reservations.get("after-oversized")!);
governor.release(reservations.get("preparation")!);
assert.equal(governor.stats().committedBytes, 0);

const pressure = governor.reserve({ owner: "source-plan", generation: "1", priority: "visible",
  claims: [visualizationReservationClaim("source-plan", "native-read", 200)] });
assert(pressure.ok);
schedule("source-tile", 60);
assert.equal(reservations.has("source-tile"), true, "source planning cannot deadlock the tile reads that fulfill it");
governor.release(pressure.reservation.id);
await flush();
governor.release(reservations.get("source-tile")!);
assert.equal(governor.stats().committedBytes, 0);
assert.throws(() => new VisualizationStagingQueue(governor, 0));
const limited = new VisualizationResourceGovernor({ totalBytes: 32 });
const limitedQueue = new VisualizationStagingQueue(limited, 128);
let rejected = false;
limitedQueue.schedule({ owner: "too-large", generation: "1", priority: "visible",
  claims: [visualizationReservationClaim("too-large", "handoff", 64)] },
  (outcome) => { rejected = !outcome.ok; });
assert(rejected);
assert.equal(limited.stats().committedBytes, 0);

const prioritizedGovernor = new VisualizationResourceGovernor();
const prioritizedQueue = new VisualizationStagingQueue(prioritizedGovernor, 100);
const priorityReservations = new Map<string, number>();
const schedulePriority = (name: string, bytes: number, priority: "interactive" | "background") => prioritizedQueue.schedule({
  owner: name, generation: "1", priority,
  claims: [visualizationReservationClaim(name, "native-read", bytes)],
}, outcome => {
  assert(outcome.ok);
  priorityReservations.set(name, outcome.reservation.id);
});
schedulePriority("running", 80, "background");
schedulePriority("prefetch", 80, "background");
schedulePriority("selected-frame", 20, "interactive");
assert.deepEqual([...priorityReservations.keys()], ["running", "selected-frame"],
  "a queued speculative tile cannot hide a fitting visible-frame read");
prioritizedQueue.setMaximumBytes(200);
assert(priorityReservations.has("prefetch"), "growing the local budget releases queued reservations");
prioritizedQueue.setMaximumBytes(10);
schedulePriority("after-pressure", 5, "interactive");
assert(!priorityReservations.has("after-pressure"), "shrinking staging waits for active commitments to drain");
for (const name of ["running", "prefetch", "selected-frame"]) prioritizedGovernor.release(priorityReservations.get(name)!);
await flush();
assert(priorityReservations.has("after-pressure"));
prioritizedGovernor.release(priorityReservations.get("after-pressure")!);
assert.equal(prioritizedGovernor.stats().committedBytes, 0);
console.log("Visualization staging queue: passed");
