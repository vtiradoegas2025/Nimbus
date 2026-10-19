/*
 * Tests for `src/lib/plot/viewSync.ts` — the pure state transitions that keep a
 * plot's scale stable across frames and its cameras stable across panels.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types \
 *     --import ../tests/ui/register-ts-resolve.mjs ../tests/ui/unit/plot/test_view_sync.ts
 *
 * These were extracted from the components precisely because they were the parts
 * that went wrong: a reset that landed one frame late, and a camera broadcast
 * that echoed itself. Both defects are pinned here, along with the render-loop
 * properties (identity stability, convergence) that make them safe in React.
 */

import {
  cameraSignature,
  carriedMagnitude,
  carriedRange,
  latchMagnitude,
  latchRange,
  planeViewportFromRelayout,
  planeViewportForScope,
  type LatchedMagnitude,
  type LatchedRange,
} from "../../../../ui/src/lib/plot/viewSync.ts";

let passed = 0;
let failed = 0;
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

// ---------- latchRange ----------

check("latchRange widens monotonically within a scope", () => {
  let state: LatchedRange = { scope: "run/w", range: null };
  state = latchRange(state, "run/w", { min: -2, max: 5 });
  assertTrue(state.range?.min === -2 && state.range?.max === 5, "first frame seeds the range");
  state = latchRange(state, "run/w", { min: -1, max: 3 });
  assertTrue(state.range?.min === -2 && state.range?.max === 5, "a narrower frame does not shrink it");
  state = latchRange(state, "run/w", { min: -9, max: 12 });
  assertTrue(state.range?.min === -9 && state.range?.max === 12, "a wider frame widens both ends");
});

check("latchRange discards the range when the scope changes", () => {
  const held: LatchedRange = { scope: "run/w", range: { min: -40, max: 40 } };
  assertTrue(carriedRange(held, "run/qc") === null, "nothing is carried into a new scope");
  const next = latchRange(held, "run/qc", { min: 0, max: 0.002 });
  assertTrue(
    next.scope === "run/qc" && next.range?.min === 0 && next.range?.max === 0.002,
    "the new scope starts from its own frame, not the old ±40",
  );
});

/* The defect this file exists for. The scale a frame is DRAWN with is resolved
 * during render from (state, scope, frame). Resetting the latch from an effect
 * instead would leave the first frame of a new scope already drawn against the
 * old scope's range, because effects run after the render they belong to. */
check("latchRange never lets a stale scope's range reach a rendered frame", () => {
  const drawn: Array<number | undefined> = [];
  let state: LatchedRange = { scope: "run/w", range: null };
  const renderFrame = (scope: string, frame: { min: number; max: number }) => {
    const resolved = latchRange(state, scope, frame).range;   // render-phase
    drawn.push(resolved?.max);
    state = latchRange(state, scope, frame);                  // effect
  };
  renderFrame("run/w", { min: -50, max: 50 });
  renderFrame("run/w", { min: -12, max: 12 });
  renderFrame("run/qc", { min: 0, max: 0.002 });             // scope change
  assertTrue(drawn[1] === 50, "within a scope the held maximum is used");
  assertTrue(
    drawn[2] === 0.002,
    "the new scope's first RENDERED frame uses its own bounds, not the stale 50",
  );
});

check("latchRange is identity-stable so a React setState bails out", () => {
  const state: LatchedRange = { scope: "run/w", range: { min: -2, max: 5 } };
  assertTrue(
    latchRange(state, "run/w", { min: -1, max: 4 }) === state,
    "a frame inside the held range returns the SAME object (no re-render)",
  );
  assertTrue(
    latchRange(state, "run/w", null) === state,
    "a null frame returns the same object",
  );
  assertTrue(
    latchRange(state, "run/w", { min: -2, max: 5 }) === state,
    "an identical frame returns the same object",
  );
  // Convergence: re-applying the transition to its own output is a fixed point,
  // which is what makes the render -> effect -> render cycle terminate.
  const widened = latchRange(state, "run/w", { min: -9, max: 9 });
  assertTrue(
    latchRange(widened, "run/w", { min: -9, max: 9 }) === widened,
    "the transition reaches a fixed point after one step",
  );
});

check("latchRange handles the empty latch and null frames", () => {
  const empty: LatchedRange = { scope: "s", range: null };
  assertTrue(latchRange(empty, "s", null) === empty, "null frame on an empty latch is a no-op");
  const seeded = latchRange(empty, "s", { min: 1, max: 2 });
  assertTrue(seeded.range?.min === 1, "a frame seeds an empty latch");
  assertTrue(
    latchRange(seeded, "s", null).range?.max === 2,
    "a later null frame keeps what was already held",
  );
});

// ---------- latchMagnitude ----------

check("latchMagnitude rises but never falls within a scope", () => {
  let state: LatchedMagnitude = { scope: "run/u,v/2", reference: 0 };
  state = latchMagnitude(state, "run/u,v/2", 5);
  assertTrue(state.reference === 5, "first frame seeds the reference");
  state = latchMagnitude(state, "run/u,v/2", 2);
  assertTrue(state.reference === 5, "a weaker frame does not lower the reference");
  state = latchMagnitude(state, "run/u,v/2", 50);
  assertTrue(state.reference === 50, "a stronger frame raises it");
});

/* The VectorField defect: switching component pair or plane must not draw one
 * frame at the previous pair's scale. A 50 m/s vertical pair followed by a
 * 5 m/s horizontal pair would otherwise render arrows ten times too short. */
check("latchMagnitude never lets a stale scope's reference reach a rendered frame", () => {
  const drawn: number[] = [];
  let state: LatchedMagnitude = { scope: "run/w,u/1", reference: 0 };
  const renderFrame = (scope: string, frameMax: number) => {
    drawn.push(Math.max(carriedMagnitude(state, scope), frameMax));  // render-phase
    state = latchMagnitude(state, scope, frameMax);                  // effect
  };
  renderFrame("run/w,u/1", 50);
  renderFrame("run/w,u/1", 12);
  renderFrame("run/u,v/2", 5);                                       // scope change
  assertTrue(drawn[1] === 50, "within a scope the held peak is used");
  assertTrue(drawn[2] === 5, "the new scope's first RENDERED frame uses its own peak, not 50");
});

check("latchMagnitude is identity-stable and re-anchors a carried scope", () => {
  const state: LatchedMagnitude = { scope: "a", reference: 9 };
  assertTrue(latchMagnitude(state, "a", 4) === state, "a weaker frame returns the same object");
  assertTrue(latchMagnitude(state, "a", 9) === state, "an equal frame returns the same object");
  const moved = latchMagnitude(state, "b", 0);
  assertTrue(moved.scope === "b" && moved.reference === 0, "a new scope re-anchors at zero");
  assertTrue(latchMagnitude(moved, "b", 0) === moved, "and then holds steady");
});

// ---------- plane viewport relayout ----------

check("plane viewport belongs to one physical coordinate view", () => {
  const bounds = { xMin: -1, xMax: 1, yMin: 0, yMax: 2 };
  const plan = { scope: "run/w/z", viewport: bounds };
  assertTrue(planeViewportForScope(plan, "run/w/z") === bounds, "frame updates retain the selected viewport");
  assertTrue(planeViewportForScope(plan, "run/w/x") === null, "x section starts with its complete vertical extent");
  assertTrue(planeViewportForScope(plan, "run/w/y") === null, "y section starts with its complete vertical extent");
  assertTrue(planeViewportForScope(plan, "other/w/z") === null, "another source cannot inherit physical bounds");
});

check("plane relayout normalizes reversed ranges", () => {
  const viewport = planeViewportFromRelayout({
    "xaxis.range[0]": 12,
    "xaxis.range[1]": 2,
    "yaxis.range": [9, 3],
  }, null);
  assertTrue(viewport?.xMin === 2 && viewport.xMax === 12, "x range is ascending");
  assertTrue(viewport?.yMin === 3 && viewport.yMax === 9, "y range is ascending");
});

check("plane relayout carries the untouched axis", () => {
  const previous = { xMin: 0, xMax: 10, yMin: 20, yMax: 30 };
  const viewport = planeViewportFromRelayout({
    "xaxis.range": [2, 4],
  }, previous);
  assertTrue(viewport?.xMin === 2 && viewport.xMax === 4, "changed axis advances");
  assertTrue(viewport?.yMin === 20 && viewport.yMax === 30, "unchanged axis is retained");
});

check("plane autorange clears the source viewport", () => {
  assertTrue(planeViewportFromRelayout({ "xaxis.autorange": true }, {
    xMin: 0, xMax: 1, yMin: 0, yMax: 1,
  }) === null, "autorange returns to the full-plane overview");
});

// ---------- cameraSignature ----------

const cam = (x: number, y = 1, z = 1) => ({
  eye: { x, y, z },
  center: { x: 0, y: 0, z: 0 },
  up: { x: 0, y: 0, z: 1 },
});

check("cameraSignature distinguishes value, not object identity", () => {
  assertTrue(
    cameraSignature(cam(2)) === cameraSignature(cam(2)),
    "two distinct objects with equal values share a signature",
  );
  assertTrue(cameraSignature(cam(2)) !== cameraSignature(cam(3)), "a moved eye differs");
  assertTrue(
    cameraSignature({ ...cam(2), up: { x: 1, y: 0, z: 0 } }) !== cameraSignature(cam(2)),
    "a rolled up-vector differs",
  );
});

check("cameraSignature returns empty for non-cameras", () => {
  for (const value of [null, undefined, 0, "eye", [], {}, { eye: 3 }]) {
    assertTrue(cameraSignature(value) === "", `${JSON.stringify(value) ?? "undefined"} has no signature`);
  }
  // A partial camera is still a camera — Plotly may omit `up`.
  assertTrue(cameraSignature({ eye: { x: 1, y: 2, z: 3 } }) !== "", "eye alone is signable");
});

/* The ComparisonVolumePlot defect: broadcasting a camera makes each sibling
 * scene emit its own relayout carrying that camera back. Acting on every event
 * re-broadcasts forever. */
check("cameraSignature lets a handler swallow the echo but not a real orbit", () => {
  let signature = "";
  let broadcasts = 0;
  const onRelayout = (camera: unknown) => {
    const key = cameraSignature(camera);
    if (!key || key === signature) return;
    signature = key;
    broadcasts++;
  };
  onRelayout(cam(2));                       // the user orbits one scene
  onRelayout(cam(2));                       // echo from a sibling (fresh object)
  onRelayout(cam(2));                       // echo from the other sibling
  assertTrue(broadcasts === 1, "one drag plus its two echoes broadcasts exactly once");
  onRelayout(cam(3));
  assertTrue(broadcasts === 2, "a genuinely new camera still broadcasts");
  onRelayout(null);
  assertTrue(broadcasts === 2, "a relayout without a camera is ignored");
});

// ---------- Report ----------

console.log(`\nviewSync tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  " + f);
  process.exit(1);
}
process.exit(0);
