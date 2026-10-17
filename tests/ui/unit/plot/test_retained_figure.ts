/*
 * Tests for `src/lib/plot/retainedFigure.ts` — the rule that keeps a figure on screen through a
 * load gap so the Plotly canvas updates in place instead of being torn down and rebuilt.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types --import ../tests/ui/register-ts-resolve.mjs ../tests/ui/unit/plot/test_retained_figure.ts
 */

import {
  selectRetainedFigure,
  type RetainedFigure,
} from "../../../../ui/src/lib/plot/retainedFigure.ts";

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

/** Replays a sequence of renders the way the hook does, returning what each one showed. */
function replay<T>(steps: ReadonlyArray<{ current: T | null; scope: string }>): Array<{
  shown: T | null;
  stale: boolean;
}> {
  let retained: RetainedFigure<T> | null = null;
  return steps.map(({ current, scope }) => {
    const selection = selectRetainedFigure(current, retained, scope);
    retained = selection.retain;
    return { shown: selection.shown, stale: selection.stale };
  });
}

// ---------- The basic rule ----------

check("a freshly built figure is shown and is not stale", () => {
  const result = selectRetainedFigure("frame-1", null, "run|u,v|z");
  assertTrue(result.shown === "frame-1", "shows the current figure");
  assertTrue(!result.stale, "not stale");
  assertTrue(result.retain?.figure === "frame-1", "carried forward");
});

check("nothing to show and nothing held renders nothing", () => {
  const result = selectRetainedFigure(null, null, "run|u,v|z");
  assertTrue(result.shown === null, "no figure");
  assertTrue(!result.stale, "not stale - there is no previous picture");
  assertTrue(result.retain === null, "nothing carried");
});

// ---------- The defect this exists to prevent ----------

check("a load gap inside one scope keeps the previous figure on screen", () => {
  // The two components of a vector plot resolve one at a time, so the coherence gate nulls the
  // figure between them. That null used to unmount the canvas rather than blank it.
  const frames = replay([
    { current: "frame-1", scope: "run|u,v|z" },
    { current: null, scope: "run|u,v|z" },
    { current: null, scope: "run|u,v|z" },
    { current: "frame-2", scope: "run|u,v|z" },
  ]);
  assertTrue(frames.every((frame) => frame.shown !== null), "the canvas never loses its figure");
  assertTrue(frames[1].shown === "frame-1" && frames[1].stale, "gap shows the previous frame");
  assertTrue(frames[2].shown === "frame-1" && frames[2].stale, "a longer gap holds too");
  assertTrue(frames[3].shown === "frame-2" && !frames[3].stale, "the new frame takes over");
});

check("playback never blanks between frames", () => {
  const steps: Array<{ current: string | null; scope: string }> = [];
  for (let frame = 0; frame < 10; frame++) {
    steps.push({ current: null, scope: "run|u,v|z" });
    steps.push({ current: `frame-${frame}`, scope: "run|u,v|z" });
  }
  const shown = replay(steps);
  // The very first render has nothing to hold; every render after it must show something.
  assertTrue(shown[0].shown === null, "cold start has no figure yet");
  assertTrue(shown.slice(1).every((frame) => frame.shown !== null), "no blank after the first draw");
});

// ---------- Scope is what keeps retention honest ----------

check("a scope change drops the held figure rather than mislabelling it", () => {
  // Switching the slice axis re-picks a component; showing the old plane under the new controls
  // would present one view's picture as another's.
  const frames = replay([
    { current: "z-plane", scope: "run|u,v|z" },
    { current: null, scope: "run|v,w|y" },
  ]);
  assertTrue(frames[1].shown === null, "the y-plane does not borrow the z-plane figure");
  assertTrue(!frames[1].stale, "and does not claim to be mid-transition");
});

check("a figure from an abandoned scope cannot come back", () => {
  const frames = replay([
    { current: "z-plane", scope: "run|u,v|z" },
    { current: null, scope: "run|v,w|y" },
    { current: null, scope: "run|u,v|z" },
  ]);
  assertTrue(frames[2].shown === null, "returning to a scope does not resurrect its old figure");
});

check("a different run never shows the previous run's figure", () => {
  const frames = replay([
    { current: "run-a-frame", scope: "run-a|u,v|z" },
    { current: null, scope: "run-b|u,v|z" },
  ]);
  assertTrue(frames[1].shown === null, "no cross-run carry-over");
});

// ---------- Value handling ----------

check("undefined is treated as absent, not as a figure", () => {
  const result = selectRetainedFigure(undefined, { scope: "s", figure: "held" }, "s");
  assertTrue(result.shown === "held", "holds through undefined");
  assertTrue(result.stale, "and reports the transition");
});

check("a falsy but present figure is still a figure", () => {
  // Guarding on truthiness rather than nullishness would discard a legitimate empty figure.
  const result = selectRetainedFigure(0, { scope: "s", figure: 42 }, "s");
  assertTrue(result.shown === 0, "zero is shown");
  assertTrue(!result.stale, "not a transition");
  assertTrue(result.retain?.figure === 0, "and is what gets carried");
});

// ---------- Summary ----------

for (const line of failures) console.error(line);
console.log(`\nretained-figure tests: ${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
