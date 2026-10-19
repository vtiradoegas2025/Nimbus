/*
 * Tests for `src/lib/plot/playbackGate.ts` and `reportsFrameIdentity` —
 * the rule that decides when frame playback may advance in a multi-pane view.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types --import ../tests/ui/register-ts-resolve.mjs ../tests/ui/unit/plot/test_playback_gate.ts
 */

import {
  EMPTY_PRESENTATION_DWELL,
  panesAreLoadingFrame,
  panesReadyForAdvance,
  paneHasDrawnFrame,
  updatePresentationDwell,
  visiblePaneTypes,
  type GatedPane,
  type PaneFrameReport,
  type PlaybackView,
} from "../../../../ui/src/lib/plot/playbackGate.ts";
import { PLOT_TYPE_META, reportsFrameIdentity, type PlotType } from "../../../../ui/src/lib/plot/plotRegistry.ts";

// ---------- Tiny assertion API (matches test_pane_layout.ts) ----------

let failed = 0;
let passed = 0;
const failures: string[] = [];

function assertTrue(cond: boolean, label: string): void {
  if (cond) passed++;
  else {
    failed++;
    failures.push(`FAIL ${label}`);
  }
}

function check(label: string, fn: () => void): void {
  try {
    fn();
  } catch (e) {
    failed++;
    failures.push(`THREW ${label}: ${(e as Error).message}`);
  }
}

// ---------- Fixtures ----------

const VIEW: PlaybackView = {
  runDir: "/runs/supercell",
  runPublicationGeneration: "gen-a",
  openEpoch: 3,
};

const SELECTED = { stepIndex: 7, publicationGeneration: "gen-a" };

function drew(plotType: PlotType, stepIndex: number, over: Partial<PaneFrameReport> = {}): PaneFrameReport {
  return {
    runDir: VIEW.runDir as string,
    runPublicationGeneration: VIEW.runPublicationGeneration,
    openEpoch: VIEW.openEpoch,
    plotType,
    requestedStepIndex: 7,
    requestedPublicationGeneration: "gen-a",
    renderedStepIndex: stepIndex,
    renderedPublicationGeneration: "gen-a",
    loading: false,
    ...over,
  };
}

const ready = (t: PlotType): GatedPane => ({ plotType: t, report: drew(t, 7) });
const behind = (t: PlotType): GatedPane => ({ plotType: t, report: drew(t, 6) });

// ---------- Which panes are on screen ----------

check("single layout gates on the primary alone", () => {
  const panes = visiblePaneTypes("single", "slice", ["volume", "isosurface", "streamlines"]);
  assertTrue(panes.length === 1, "one pane");
  assertTrue(panes[0] === "slice", "the primary");
});

check("a grid gates on the primary plus exactly its companions", () => {
  const companions: PlotType[] = ["volume", "isosurface", "streamlines"];
  assertTrue(
    JSON.stringify(visiblePaneTypes("1x2", "slice", companions)) === JSON.stringify(["slice", "volume"]),
    "1x2 takes one companion",
  );
  assertTrue(
    JSON.stringify(visiblePaneTypes("2x1", "slice", companions)) === JSON.stringify(["slice", "volume"]),
    "2x1 takes one companion",
  );
  assertTrue(
    JSON.stringify(visiblePaneTypes("2x2", "slice", companions))
      === JSON.stringify(["slice", "volume", "isosurface", "streamlines"]),
    "2x2 gates the primary plus all three companions",
  );
});

check("companions beyond the layout cannot hold playback", () => {
  // Shrinking 2x2 -> 1x2 leaves two selections behind; neither is drawn, so neither is waited on.
  const panes = visiblePaneTypes("1x2", "slice", ["volume", "isosurface", "streamlines"]);
  assertTrue(!panes.includes("isosurface"), "dropped companion is not gated");
  assertTrue(!panes.includes("streamlines"), "second dropped companion is not gated");
});

// ---------- Per-pane readiness ----------

check("a pane that drew the selected frame is ready", () => {
  assertTrue(paneHasDrawnFrame(ready("slice"), VIEW, SELECTED), "slice at step 7");
});

check("a pane still on the previous frame is not ready", () => {
  assertTrue(!paneHasDrawnFrame(behind("slice"), VIEW, SELECTED), "slice still at step 6");
});

check("a pane that has drawn nothing is not ready", () => {
  assertTrue(
    !paneHasDrawnFrame({ plotType: "slice", report: null }, VIEW, SELECTED),
    "no report is not ready",
  );
  assertTrue(
    !paneHasDrawnFrame({ plotType: "volume", report: undefined }, VIEW, SELECTED),
    "absent map entry is not ready",
  );
});

check("a pane still loading the selected frame is not ready", () => {
  assertTrue(
    !paneHasDrawnFrame({ plotType: "slice", report: drew("slice", 7, { loading: true }) }, VIEW, SELECTED),
    "rendered but still loading",
  );
});

check("active loading is distinguished from a missing or failed report", () => {
  assertTrue(
    panesAreLoadingFrame(
      [{ plotType: "isosurface", report: drew("isosurface", 6, { loading: true }) }],
      VIEW,
      SELECTED,
    ),
    "selected high-resolution frame is actively loading",
  );
  assertTrue(
    !panesAreLoadingFrame([behind("isosurface")], VIEW, SELECTED),
    "a non-loading stale frame is not active work",
  );
  assertTrue(
    !panesAreLoadingFrame(
      [{
        plotType: "isosurface",
        report: drew("isosurface", 6, {
          loading: true,
          requestedStepIndex: 8,
        }),
      }],
      VIEW,
      SELECTED,
    ),
    "loading a different frame cannot extend this frame's hold",
  );
});

check("a report from another view is not accepted", () => {
  const cases: Array<[string, Partial<PaneFrameReport>]> = [
    ["another run", { runDir: "/runs/other" }],
    ["another publication generation", { runPublicationGeneration: "gen-b" }],
    ["another open epoch", { openEpoch: 4 }],
  ];
  for (const [label, over] of cases) {
    assertTrue(
      !paneHasDrawnFrame({ plotType: "slice", report: drew("slice", 7, over) }, VIEW, SELECTED),
      `stale report rejected: ${label}`,
    );
  }
});

check("a report left by the pane's previous plot type is not accepted", () => {
  // The pane now shows a volume; the ready report belongs to the slice it used to show.
  assertTrue(
    !paneHasDrawnFrame({ plotType: "volume", report: drew("slice", 7) }, VIEW, SELECTED),
    "type mismatch rejected",
  );
});

check("a frame rendered under a different generation is not accepted", () => {
  assertTrue(
    !paneHasDrawnFrame(
      { plotType: "slice", report: drew("slice", 7, { renderedPublicationGeneration: "gen-b" }) },
      VIEW,
      SELECTED,
    ),
    "step number alone is not identity",
  );
});

// ---------- Types that cannot report must never deadlock ----------

check("a non-reporting type is ready even with no report at all", () => {
  const nonReporting = PLOT_TYPE_META
    .map((meta) => meta.value)
    .filter((value) => !reportsFrameIdentity(value));
  assertTrue(nonReporting.length > 0, "there are non-reporting plot types");
  for (const plotType of nonReporting) {
    assertTrue(
      paneHasDrawnFrame({ plotType, report: null }, VIEW, SELECTED),
      `${plotType} cannot stall playback`,
    );
  }
});

check("every reporting type is a real plot type", () => {
  const known = new Set(PLOT_TYPE_META.map((meta) => meta.value));
  for (const plotType of known) {
    if (!reportsFrameIdentity(plotType)) continue;
    assertTrue(known.has(plotType), `${plotType} is registered`);
  }
  assertTrue(reportsFrameIdentity("slice"), "slice reports");
  assertTrue(reportsFrameIdentity("volume"), "volume reports");
  assertTrue(reportsFrameIdentity("isosurface"), "isosurface reports");
  assertTrue(reportsFrameIdentity("streamlines"), "streamlines report");
  assertTrue(reportsFrameIdentity("supercell"), "supercell reports");
});

check("no selected frame means nothing to wait for", () => {
  assertTrue(
    paneHasDrawnFrame({ plotType: "slice", report: null }, VIEW, null),
    "an empty run does not stall",
  );
});

// ---------- The grid decision ----------

check("a grid advances only when every reporting pane has drawn the frame", () => {
  assertTrue(
    panesReadyForAdvance([ready("slice"), ready("volume")], VIEW, SELECTED),
    "both panes current",
  );
  assertTrue(
    !panesReadyForAdvance([ready("slice"), behind("volume")], VIEW, SELECTED),
    "a lagging companion holds the grid",
  );
  assertTrue(
    !panesReadyForAdvance([behind("slice"), ready("volume")], VIEW, SELECTED),
    "a lagging primary holds the grid",
  );
});

check("this is the defect the gate exists to prevent", () => {
  // Before the fix only the primary was consulted, so a ready primary advanced the timer while a
  // companion was still two frames back - each pane animating at its own speed.
  const grid = [ready("slice"), behind("volume"), behind("isosurface")];
  assertTrue(paneHasDrawnFrame(grid[0], VIEW, SELECTED), "primary alone would have advanced");
  assertTrue(!panesReadyForAdvance(grid, VIEW, SELECTED), "the grid as a whole waits");
});

check("a grid of non-reporting panes advances on the timer", () => {
  assertTrue(
    panesReadyForAdvance(
      [{ plotType: "radial", report: null }, { plotType: "hodograph", report: null }],
      VIEW,
      SELECTED,
    ),
    "nothing to gate on",
  );
});

check("an empty pane list is ready", () => {
  assertTrue(panesReadyForAdvance([], VIEW, SELECTED), "no panes, no wait");
});

// ---------- Presented-frame dwell ----------

check("dwell starts only after the exact frame is ready", () => {
  const waiting = updatePresentationDwell(
    EMPTY_PRESENTATION_DWELL,
    SELECTED,
    false,
    1_000,
    500,
  );
  assertTrue(waiting.state.readySinceMs == null, "loads do not consume visible dwell");
  assertTrue(!waiting.advance, "an unready frame cannot advance");

  const presented = updatePresentationDwell(waiting.state, SELECTED, true, 1_800, 500);
  assertTrue(presented.state.readySinceMs === 1_800, "dwell begins at presentation");
  assertTrue(!presented.advance, "new presentation remains visible");
});

check("a coherent frame advances only after a full visible dwell", () => {
  const presented = updatePresentationDwell(
    EMPTY_PRESENTATION_DWELL,
    SELECTED,
    true,
    2_000,
    500,
  );
  const early = updatePresentationDwell(presented.state, SELECTED, true, 2_499, 500);
  assertTrue(!early.advance, "one millisecond short still holds");
  const due = updatePresentationDwell(early.state, SELECTED, true, 2_500, 500);
  assertTrue(due.advance, "full dwell advances");
});

check("changing identity restarts dwell", () => {
  const first = updatePresentationDwell(
    EMPTY_PRESENTATION_DWELL,
    SELECTED,
    true,
    3_000,
    500,
  );
  const next = updatePresentationDwell(
    first.state,
    { stepIndex: 8, publicationGeneration: "gen-a" },
    true,
    3_700,
    500,
  );
  assertTrue(!next.advance, "new frame does not inherit old dwell");
  assertTrue(next.state.readySinceMs === 3_700, "new frame gets its own clock");
});

// ---------- Summary ----------

for (const line of failures) console.error(line);
console.log(`\nplayback-gate tests: ${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
