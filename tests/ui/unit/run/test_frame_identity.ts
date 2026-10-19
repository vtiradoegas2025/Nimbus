/*
 * Deterministic requested/loaded/rendered identity tests.
 * Run via: npm run test:frames
 */

import {
  beginFrameRequest,
  captureFrameIsReady,
  completeFrameRequest,
  describeFrameTransition,
  findPublishedFrame,
  loadedFramesAreCoherent,
  runViewIdentityKey,
} from "../../../../ui/src/lib/run/frameIdentity.ts";
import { captureFrameIndices } from "../../../../ui/src/lib/run/frameReady.ts";

let passed = 0;
let failed = 0;
const failures: string[] = [];

function equal<T>(actual: T, expected: T, label: string): void {
  if (actual === expected) passed++;
  else {
    failed++;
    failures.push(`FAIL ${label}: expected ${String(expected)}, got ${String(actual)}`);
  }
}

const retained = beginFrameRequest(10, "generation-a", 20, "generation-a", true);
equal(retained.requestedStepIndex, 20, "request advances immediately");
equal(retained.loadedStepIndex, 10, "same series retains old loaded identity");

const switched = beginFrameRequest(10, "generation-a", 20, "generation-a", false);
equal(switched.loadedStepIndex, null, "new run/field clears loaded identity");

const completed = completeFrameRequest(20, "generation-a");
equal(completed.loadedStepIndex, 20, "completion records exact loaded identity");

const stale = describeFrameTransition(20, 10, "generation-a", "generation-a");
equal(stale.kind, "stale", "old pixels are marked stale");
equal(stale.label, "Showing step 10 while step 20 loads", "stale label names both steps");

equal(
  describeFrameTransition(20, null, "generation-a", null).kind,
  "loading",
  "no pixels is loading",
);
equal(
  describeFrameTransition(20, 20, "generation-a", "generation-a").kind,
  "ready",
  "matching pixels and generation are ready",
);
equal(
  describeFrameTransition(null, null, null, null).kind,
  "empty",
  "no request is empty",
);

const reopened = beginFrameRequest(
  20,
  "generation-a",
  20,
  "generation-b",
  false,
);
equal(reopened.loadedStepIndex, null, "same-dir/same-step reopen clears old loaded step");
equal(
  reopened.loadedPublicationGeneration,
  null,
  "same-dir/same-step reopen clears old loaded generation",
);
equal(
  describeFrameTransition(20, 20, "generation-b", "generation-a").kind,
  "stale",
  "matching step numbers from different generations are never ready",
);
equal(
  runViewIdentityKey("/same/run", "generation-a", 1) ===
    runViewIdentityKey("/same/run", "generation-b", 2),
  false,
  "same-directory current-run reopen remounts the plot subtree",
);
equal(
  runViewIdentityKey("/legacy/run", null, 1) ===
    runViewIdentityKey("/legacy/run", null, 2),
  false,
  "same-directory legacy reopen remounts via the open epoch",
);

equal(
  captureFrameIsReady(
    { stepIndex: 20, publicationGeneration: "generation-b" },
    {
      renderedStepIndex: 20,
      renderedPublicationGeneration: "generation-b",
      loading: false,
    },
  ),
  true,
  "recorder accepts the exact rendered absolute frame identity",
);
equal(
  captureFrameIsReady(
    { stepIndex: 20, publicationGeneration: "generation-b" },
    {
      renderedStepIndex: 20,
      renderedPublicationGeneration: "generation-a",
      loading: false,
    },
  ),
  false,
  "recorder rejects a reused step from an older publication",
);
equal(
  captureFrameIsReady(
    { stepIndex: 20, publicationGeneration: "generation-b" },
    {
      renderedStepIndex: 20,
      renderedPublicationGeneration: "generation-b",
      loading: true,
    },
  ),
  false,
  "recorder rejects a matching frame until rendering settles",
);

equal(
  loadedFramesAreCoherent(
    { loadedStepIndex: 20, loadedPublicationGeneration: "generation-b" },
    { loadedStepIndex: 20, loadedPublicationGeneration: "generation-b" },
  ),
  true,
  "multi-field render accepts one exact publication",
);
equal(
  loadedFramesAreCoherent(
    { loadedStepIndex: 20, loadedPublicationGeneration: "generation-b" },
    { loadedStepIndex: 10, loadedPublicationGeneration: "generation-b" },
  ),
  false,
  "multi-field render rejects torn numeric steps",
);
equal(
  loadedFramesAreCoherent(
    { loadedStepIndex: 20, loadedPublicationGeneration: "generation-a" },
    { loadedStepIndex: 20, loadedPublicationGeneration: "generation-b" },
  ),
  false,
  "multi-field render rejects reused steps from different generations",
);
equal(
  loadedFramesAreCoherent(
    { loadedStepIndex: null, loadedPublicationGeneration: null },
  ),
  false,
  "multi-field render rejects an unresolved input",
);

const publications = [
  { index: 20, publicationGeneration: "generation-a", marker: "old" },
  { index: 20, publicationGeneration: "generation-b", marker: "new" },
];
equal(
  findPublishedFrame(publications, 20, "generation-b")?.marker,
  "new",
  "loaded metadata lookup matches generation as well as step",
);
equal(
  findPublishedFrame(publications, 20, "generation-c"),
  null,
  "loaded metadata lookup never falls back to another generation",
);

equal(
  JSON.stringify(captureFrameIndices(25, 0, 24, 5)),
  JSON.stringify([0, 5, 10, 15, 20]),
  "animation export strides absolute solved frames",
);
equal(
  JSON.stringify(captureFrameIndices(25, 3, 9, 2)),
  JSON.stringify([3, 5, 7, 9]),
  "animation export preserves the selected start and inclusive aligned end",
);
equal(
  captureFrameIndices(0, 0, 10, 1).length,
  0,
  "animation export handles an empty run",
);

console.log(`\nframeIdentity tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const failure of failures) console.log(`  ${failure}`);
  process.exit(1);
}
