/*
 * Playback frame prefetch.
 *
 * Playback will not advance until the current frame has rendered, so without
 * warming the loop is strictly serial — read, render, read, render — and every
 * step pays a full cold read no matter what speed is selected.
 */

import {
  MAX_INFLIGHT_PREFETCH,
  PREFETCH_DEPTH,
  endPrefetch,
  inflightPrefetchCount,
  tryBeginPrefetch,
  framesToPrefetch,
  registerPlaybackPrefetchAlias,
  playbackPrefetchHint,
  setPlaybackPrefetchHint,
  upcomingSteps,
  createPrefetchController,
  releasePrefetchController,
  clearPlaybackFrameMetadata,
  registerPlaybackFrameMetadata,
  frameIdentitiesToPrefetch,
  playbackFrameMetadataCount,
} from "../../../../ui/src/lib/run/framePrefetch.ts";
import { fieldLoadScheduler } from "../../../../ui/src/lib/core/loadScheduler.ts";

let passed = 0;
const failures: string[] = [];
const check = (condition: boolean, label: string): void => {
  if (condition) passed++;
  else failures.push(`FAIL ${label}`);
};

const frames = (indices: number[]) => indices.map((index) => ({ index }));

/* ---------- upcomingSteps ---------- */

check(
  upcomingSteps(frames([0, 10, 20, 30]), 0).join(",") === "10",
  "warms the next frame",
);
check(
  upcomingSteps(frames([0, 10, 20, 30]), 0).length === PREFETCH_DEPTH,
  "warms exactly the configured depth",
);
// Playback loops, so the frames after the last one are the first ones.
check(
  upcomingSteps(frames([0, 10, 20, 30]), 3).join(",") === "0",
  "wraps at the end because playback loops",
);
check(
  upcomingSteps(frames([0, 10, 20, 30]), 2).join(",") === "30",
  "wraps mid-list without skipping a frame",
);
// Sparse simulation output: absolute step numbers, not array positions.
check(
  upcomingSteps(frames([0, 30, 60, 90]), 1).join(",") === "60",
  "carries absolute step numbers, not array positions",
);

/* A run with too few frames must not queue the SAME frame repeatedly: that is
 * pure waste, and with a single frame there is nothing to warm at all. */
check(upcomingSteps(frames([5]), 0).length === 0, "a single-frame run warms nothing");
check(upcomingSteps([], 0).length === 0, "an empty run warms nothing");
check(
  upcomingSteps(frames([0, 10]), 0).join(",") === "10",
  "a two-frame run warms only the other frame",
);
check(
  upcomingSteps(frames([0, 10]), 1).join(",") === "0",
  "a two-frame run wraps to the first frame",
);
check(
  upcomingSteps(frames([0, 10, 20, 30, 40, 50]), 1, 2).join(",") === "30",
  "frame-stride playback warms the next frame that will actually be displayed",
);
check(
  upcomingSteps(frames([0, 10, 20, 30, 40, 50]), 5, 5).join(",") === "40",
  "frame-stride prefetch wraps without warming skipped frames",
);
check(
  upcomingSteps(frames([0, 10]), 0, 10).join(",") === "10",
  "a stride longer than the run still warms another frame",
);

/* ---------- hint scoping ---------- */

setPlaybackPrefetchHint(null);
check(playbackPrefetchHint() === null, "no hint while playback is stopped");
check(
  framesToPrefetch("/runs/a", "gen-1").length === 0,
  "nothing is warmed while playback is stopped",
);

setPlaybackPrefetchHint({
  runDir: "/runs/a",
  publicationGeneration: "gen-1",
  upcoming: [10, 20, 30, 40],
});
const releaseNestedAlias = registerPlaybackPrefetchAlias(
  "/runs/a/nests/child_0000",
  "/runs/a",
);
check(
  framesToPrefetch("/runs/a/nests/child_0000", "gen-1").join(",") === "10",
  "nested histories should reuse the root playback timeline without changing read identity",
);
releaseNestedAlias();
check(
  framesToPrefetch("/runs/a/nests/child_0000", "gen-1").length === 0,
  "releasing the last nested alias should stop root-timeline prefetch routing",
);
check(
  framesToPrefetch("/runs/a", "gen-1").join(",") === "10",
  "the matching run warms up to the depth limit",
);
check(
  framesToPrefetch("/runs/b", "gen-1").length === 0,
  "a different run is never warmed from this run's queue",
);
// A republished run reuses step numbers. Warming across generations would put
// the previous publication's bytes under the new publication's step label.
check(
  framesToPrefetch("/runs/a", "gen-2").length === 0,
  "a different publication generation is never warmed from the old queue",
);
check(
  framesToPrefetch("/runs/a", null).length === 0,
  "a legacy markerless run does not inherit a generation-scoped queue",
);

setPlaybackPrefetchHint({
  runDir: "/runs/legacy",
  publicationGeneration: null,
  upcoming: [1, 2],
});
check(
  framesToPrefetch("/runs/legacy", null).join(",") === "1",
  "a legacy run warms against its own null generation",
);
check(
  framesToPrefetch("/runs/legacy", "gen-1").length === 0,
  "a generation-bearing request does not read the legacy queue",
);

setPlaybackPrefetchHint(null);
check(
  framesToPrefetch("/runs/legacy", null).length === 0,
  "clearing the hint stops all warming",
);

/* ---------- Global ceiling ----------
 * Depth bounds one hook, but a multi-field scene mounts many at once: the
 * rendered supercell alone holds nine field hooks. Without a global ceiling
 * speculation fills the scheduler's three slots and the frame the user is
 * actually waiting for queues behind it — slower than no prefetching at all. */
check(inflightPrefetchCount() === 0, "no speculative reads outstanding at rest");

const granted: boolean[] = [];
for (let i = 0; i < MAX_INFLIGHT_PREFETCH + 5; i++) granted.push(tryBeginPrefetch());
check(
  granted.filter(Boolean).length === MAX_INFLIGHT_PREFETCH,
  `exactly ${MAX_INFLIGHT_PREFETCH} speculative reads are admitted`,
);
check(
  granted.slice(MAX_INFLIGHT_PREFETCH).every((g) => g === false),
  "requests past the ceiling are refused, not queued",
);
check(
  inflightPrefetchCount() === MAX_INFLIGHT_PREFETCH,
  "the ceiling is accounted exactly",
);

for (let i = 0; i < MAX_INFLIGHT_PREFETCH; i++) endPrefetch();
check(inflightPrefetchCount() === 0, "completed reads release their slot");
check(tryBeginPrefetch(), "slots are reusable once released");
endPrefetch();

// A double release (a settled promise plus a teardown path) must not create
// phantom capacity that lets speculation exceed the ceiling later.
endPrefetch();
endPrefetch();
check(inflightPrefetchCount() === 0, "an extra release cannot drive the count negative");

const controller = createPrefetchController();
setPlaybackPrefetchHint({ runDir: "/runs/a", publicationGeneration: "gen-a", upcoming: [1] });
setPlaybackPrefetchHint({ runDir: "/runs/a", publicationGeneration: "gen-a", upcoming: [2] });
check(!controller.signal.aborted, "a normal playback advance preserves a read that the next frame can join");
setPlaybackPrefetchHint(null);
check(controller.signal.aborted, "pausing playback cancels speculative reads");
releasePrefetchController(controller);

fieldLoadScheduler.setLimits(8, 1024 * 1024 * 1024);
const largerGrants = Array.from({ length: 8 }, () => tryBeginPrefetch());
check(largerGrants.filter(Boolean).length === 4, "a larger local reader budget warms more fields with a four-read ceiling");
for (const granted of largerGrants) if (granted) endPrefetch();
fieldLoadScheduler.setLimits(1, 64 * 1024 * 1024);
check(tryBeginPrefetch(), "a small viewer can warm one frame while otherwise idle");
check(!tryBeginPrefetch(), "a small viewer cannot spawn several speculative reads");
endPrefetch();
fieldLoadScheduler.setLimits(3, 384 * 1024 * 1024);

clearPlaybackFrameMetadata();
registerPlaybackFrameMetadata("/runs/publications", [
  { index: 0, publicationGeneration: "frame-a" },
  { index: 10, publicationGeneration: "frame-b" },
  { index: 20, publicationGeneration: "frame-c" },
], true);
setPlaybackPrefetchHint({ runDir: "/runs/publications", publicationGeneration: "frame-a", upcoming: [20] });
check(JSON.stringify(frameIdentitiesToPrefetch("/runs/publications", 10, "frame-b"))
  === JSON.stringify([{ stepIndex: 20, publicationGeneration: "frame-c" }]),
"per-frame publications warm the next frame with its own exact identity");
check(framesToPrefetch("/runs/publications", "frame-b", 10).join(",") === "20",
  "index-based readers can route a hint with a different timeline generation");
check(frameIdentitiesToPrefetch("/runs/publications", 10, "frame-old").length === 0,
  "a stale current publication cannot warm a new timeline");
check(frameIdentitiesToPrefetch("/runs/publications", 99, "frame-b").length === 0,
  "a frame missing from the registered inventory cannot speculate");
registerPlaybackFrameMetadata("/runs/publications", [{ index: 30, publicationGeneration: "frame-d" }]);
check(frameIdentitiesToPrefetch("/runs/publications", 10, "frame-b").length === 1,
  "targeted metadata updates retain other known publications");
registerPlaybackFrameMetadata("/runs/publications", [
  { index: 0, publicationGeneration: "frame-a" },
  { index: 10, publicationGeneration: "frame-b" },
], true);
check(frameIdentitiesToPrefetch("/runs/publications", 10, "frame-b").length === 0,
  "a full inventory replacement removes unavailable upcoming frames");
registerPlaybackFrameMetadata("/runs/publications", [
  { index: 0, publicationGeneration: "new-a" },
  { index: 10, publicationGeneration: "new-b" },
  { index: 20, publicationGeneration: "new-c" },
], true);
check(frameIdentitiesToPrefetch("/runs/publications", 10, "new-b").length === 0,
  "an old timeline hint is rejected after republishing");

registerPlaybackFrameMetadata("/runs/publications/nests/child", [
  { index: 10, publicationGeneration: "child-b" },
  { index: 20, publicationGeneration: "child-c" },
]);
const releasePublicationAlias = registerPlaybackPrefetchAlias("/runs/publications/nests/child", "/runs/publications");
setPlaybackPrefetchHint({ runDir: "/runs/publications", publicationGeneration: "new-a", upcoming: [20] });
check(JSON.stringify(frameIdentitiesToPrefetch("/runs/publications/nests/child", 10, "child-b"))
  === JSON.stringify([{ stepIndex: 20, publicationGeneration: "child-c" }]),
  "nested histories retain their own target publication while using the root frame order");
releasePublicationAlias();
clearPlaybackFrameMetadata();
check(playbackFrameMetadataCount() === 0 && playbackPrefetchHint() === null, "cache clearing retires frame metadata and warming hints");
for (let run = 0; run < 60; run++) {
  registerPlaybackFrameMetadata(`/runs/bounded-${run}`, Array.from({ length: 1000 }, (_, index) => ({ index, publicationGeneration: `run-${run}-frame-${index}` })));
}
check(playbackFrameMetadataCount() === 50_000, "the frame metadata bound applies across all runs, not independently per run");
clearPlaybackFrameMetadata();

console.log(`\nframePrefetch tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length > 0) {
  console.log(failures.join("\n"));
  process.exit(1);
}
