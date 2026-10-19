/*
 * Field-read cancellation scoping tests.
 * Run via: npm run test:field-cancellation
 *
 * `readFieldAny` shares one AbortController per `latestScope` lane so a newer
 * request supersedes an older one. The scope signal must be minted only when a
 * read actually starts: a cache hit or an in-flight dedup never reaches the
 * loader, so it must not abort a different frame that is still reading.
 *
 * Uses the packaged atmospheric example, which resolves without Tauri, and an
 * expression-derived field, whose evaluation honors the scope signal outside a
 * browser Worker.
 */

import { isExpressionEvaluationAbort } from "../../../../ui/src/lib/derived/expressionDerivedWorkerClient.ts";
import { createExpressionDerivedField } from "../../../../ui/src/lib/derived/expressionDerivedFields.ts";
import { PACKAGED_ATMOSPHERIC_EXAMPLES } from "../../../../ui/src/lib/dataset/packagedAtmosphericExamples.ts";
import { fieldCacheStats } from "../../../../ui/src/lib/core/fieldCache.ts";
import {
  clearFieldCache,
  listSteps,
  readFieldAny,
  readFieldWindows,
  registeredFieldRank,
  registerExpressionDerivedFields,
} from "../../../../ui/src/lib/run/runOutputs.ts";

let passed = 0;
const failures: string[] = [];
const check = (condition: boolean, label: string) => condition ? passed += 1 : failures.push(label);

const uri = PACKAGED_ATMOSPHERIC_EXAMPLES[0]!.uri;
const LANE = "plot-widget-lane";

const speed = createExpressionDerivedField({
  id: "probe_speed",
  label: "Probe speed",
  unit: "m/s",
  dependencies: ["u", "v"],
  formula: "sqrt(u^2 + v^2)",
});

clearFieldCache();
// listSteps populates the per-frame native field inventory that dependency
// resolution reads; without it `u`/`v` are not recognized as native.
const steps = await listSteps(uri);
check(steps.length >= 2, "packaged example publishes at least two frames");
const [frame0, frame1] = [steps[0]!, steps[1]!];
check(
  registeredFieldRank(uri, frame0.index, "u", frame0.publicationGeneration) === 3,
  "native catalog rank is available before the first field read",
);
// Each packaged frame carries its own generation, so the two frames are
// distinct cache keys sharing one latest-scope lane.
check(
  frame0.publicationGeneration !== frame1.publicationGeneration,
  "frames are distinguishable publications",
);

registerExpressionDerivedFields(uri, [speed]);

// Warm frame 0 so the third call below is served from cache.
await readFieldAny(uri, frame0.index, "probe_speed", frame0.publicationGeneration, { latestScope: LANE });

// Start a real read of frame 1 on the same lane, then hit the cache for frame 0.
const inFlight = readFieldAny(uri, frame1.index, "probe_speed", frame1.publicationGeneration, { latestScope: LANE });
let aborted = false;
const settled = inFlight.catch((error: unknown) => {
  aborted = isExpressionEvaluationAbort(error);
  return null;
});

const cached = await readFieldAny(uri, frame0.index, "probe_speed", frame0.publicationGeneration, { latestScope: LANE });
check(cached.data.length > 0, "cache hit returns the warmed frame");

const result = await settled;
check(!aborted, "a cache hit must not abort another frame's in-flight read");
check(result !== null && result.data.length > 0, "the superseded-looking read still resolves with data");

const transferable = await readFieldAny(
  uri,
  frame0.index,
  "probe_speed",
  frame0.publicationGeneration,
  { ownership: "transferable", cacheRetention: "transient" },
);
const transferableAgain = await readFieldAny(
  uri,
  frame0.index,
  "probe_speed",
  frame0.publicationGeneration,
  { ownership: "transferable", cacheRetention: "transient" },
);
check(
  transferable.data.buffer !== transferableAgain.data.buffer,
  "transferable reads return exclusive backing stores",
);
const transferred = structuredClone(transferable.data, {
  transfer: [transferable.data.buffer as ArrayBuffer],
});
check(transferable.data.byteLength === 0, "exclusive field storage can be detached for a worker");
check(transferred.byteLength > 0, "worker transfer preserves the field payload");
const residentAfterTransfer = await readFieldAny(
  uri,
  frame0.index,
  "probe_speed",
  frame0.publicationGeneration,
  { latestScope: LANE },
);
check(residentAfterTransfer.data.byteLength > 0, "worker transfer cannot detach resident cache storage");

const residentNative = await readFieldAny(
  uri, frame0.index, "u", frame0.publicationGeneration,
);
const transferableNative = await readFieldAny(
  uri,
  frame0.index,
  "u",
  frame0.publicationGeneration,
  { ownership: "transferable", cacheRetention: "transient" },
);
check(
  transferableNative.data.buffer !== residentNative.data.buffer,
  "native swath-style reads bypass resident backing storage",
);
structuredClone(transferableNative.data, {
  transfer: [transferableNative.data.buffer as ArrayBuffer],
});
check(transferableNative.data.byteLength === 0, "native transferable storage detaches cleanly");
check(residentNative.data.byteLength > 0, "detaching native swath storage preserves the resident field");

clearFieldCache();
await listSteps(uri);
registerExpressionDerivedFields(uri, [speed]);
const activeTransfer = readFieldAny(uri, frame0.index, "probe_speed", frame0.publicationGeneration,
  { latestScope: LANE, ownership: "transferable" }).then(value => value, () => null);
const staleController = new AbortController();
staleController.abort();
let staleRejected = false;
try {
  await readFieldAny(uri, frame1.index, "probe_speed", frame1.publicationGeneration,
    { latestScope: LANE, ownership: "transferable", signal: staleController.signal });
} catch (error) { staleRejected = isExpressionEvaluationAbort(error); }
check(staleRejected, "an already-cancelled consumer is rejected");
check((await activeTransfer) != null, "an already-cancelled read cannot supersede a live consumer on its lane");

// The lane must still supersede a genuine newer read.
clearFieldCache();
await listSteps(uri);
registerExpressionDerivedFields(uri, [speed]);
const older = readFieldAny(uri, frame0.index, "probe_speed", frame0.publicationGeneration, { latestScope: LANE });
let olderAborted = false;
const olderSettled = older.catch((error: unknown) => {
  olderAborted = isExpressionEvaluationAbort(error);
  return null;
});
const newer = await readFieldAny(uri, frame1.index, "probe_speed", frame1.publicationGeneration, { latestScope: LANE });
await olderSettled;
check(newer.data.length > 0, "the newest read on a lane resolves");
check(olderAborted, "a genuine newer read still supersedes the older one on the same lane");

clearFieldCache();
const windows = [{
  origin: [0, 0, 0], extent: [8, 6, 4], stride: [1, 1, 1], aggregation: "cell_mean" as const,
}];
const transientController = new AbortController();
const [temporaryA, temporaryB] = await Promise.all([
  readFieldWindows(uri, frame0.index, "u", frame0.publicationGeneration, windows,
    { cacheRetention: "transient", signal: transientController.signal }),
  readFieldWindows(uri, frame0.index, "u", frame0.publicationGeneration, windows,
    { cacheRetention: "transient", signal: new AbortController().signal }),
]);
check(temporaryA.data === temporaryB.data, "simultaneous transient window reads share their assembled field");
check(fieldCacheStats().bytes === 0, "transient window assemblies and tiles leave no resident cache storage");
const temporaryWithoutSignal = await readFieldWindows(
  uri, frame0.index, "u", frame0.publicationGeneration, windows, { cacheRetention: "transient" },
);
check(fieldCacheStats().bytes === 0, "transient window reads without cancellation also expire");
const retainedWindow = await readFieldWindows(uri, frame0.index, "u", frame0.publicationGeneration, windows);
check(fieldCacheStats().bytes === retainedWindow.data.byteLength, "ordinary window reads retain the final assembly once");
check(
  retainedWindow.data.every((value, index) => value === temporaryA.data[index]
    && value === temporaryWithoutSignal.data[index]),
  "temporary retention preserves every scalar value",
);
clearFieldCache();

console.log(`field read cancellation: ${passed} checks passed, ${failures.length} failed`);
if (failures.length) {
  failures.forEach((failure) => console.error(`FAIL ${failure}`));
  process.exit(1);
}
