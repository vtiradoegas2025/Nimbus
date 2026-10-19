/*
 * Tests for `src/lib/plot/paneLayout.ts` and `src/lib/plot/cameraSyncBus.ts` —
 * the pure halves of Plot Studio's multi-pane view.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types --import ../tests/ui/register-ts-resolve.mjs ../tests/ui/unit/plot/test_pane_layout.ts
 */

import {
  MAX_PANES,
  MIN_PANE_VOXEL_BUDGET,
  PANE_LAYOUTS,
  activeThreeDPaneBudgetScale,
  isPaneLayout,
  paneCount,
  paneGridTemplate,
  paneLayoutLabel,
  paneVoxelBudget,
} from "../../../../ui/src/lib/plot/paneLayout.ts";
import {
  cameraStatesEqual,
  createCameraSyncBus,
  type CameraState,
} from "../../../../ui/src/lib/plot/cameraSyncBus.ts";

// ---------- Tiny assertion API (matches test_quiver.ts) ----------

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

// ---------- Layout arithmetic ----------

check("pane counts match their layout names", () => {
  assertTrue(paneCount("single") === 1, "single -> 1");
  assertTrue(paneCount("1x2") === 2, "1x2 -> 2");
  assertTrue(paneCount("2x1") === 2, "2x1 -> 2");
  assertTrue(paneCount("2x2") === 4, "2x2 -> 4");
});

check("no layout exceeds the WebGL context ceiling", () => {
  for (const layout of PANE_LAYOUTS) {
    assertTrue(paneCount(layout) <= MAX_PANES, `${layout} within MAX_PANES`);
  }
});

check("every layout has a template and a label", () => {
  for (const layout of PANE_LAYOUTS) {
    const template = paneGridTemplate(layout);
    assertTrue(template.columns.length > 0, `${layout} has columns`);
    assertTrue(template.rows.length > 0, `${layout} has rows`);
    assertTrue(paneLayoutLabel(layout).length > 0, `${layout} has a label`);
  }
});

check("grid templates declare the track counts the layout implies", () => {
  const tracks = (value: string) => value.split(" minmax").length;
  assertTrue(tracks(paneGridTemplate("1x2").columns) === 2, "1x2 has two columns");
  assertTrue(tracks(paneGridTemplate("1x2").rows) === 1, "1x2 has one row");
  assertTrue(tracks(paneGridTemplate("2x1").columns) === 1, "2x1 has one column");
  assertTrue(tracks(paneGridTemplate("2x1").rows) === 2, "2x1 has two rows");
  assertTrue(tracks(paneGridTemplate("2x2").columns) === 2, "2x2 has two columns");
  assertTrue(tracks(paneGridTemplate("2x2").rows) === 2, "2x2 has two rows");
});

check("isPaneLayout rejects anything not in the union", () => {
  assertTrue(isPaneLayout("2x2"), "accepts a real layout");
  assertTrue(!isPaneLayout("3x3"), "rejects an unsupported grid");
  assertTrue(!isPaneLayout(""), "rejects empty");
  assertTrue(!isPaneLayout(null), "rejects null");
  assertTrue(!isPaneLayout(4), "rejects a number");
});

// ---------- Voxel budget ----------

check("the voxel budget divides across panes", () => {
  const full = 4_000_000;
  assertTrue(paneVoxelBudget(full, "single") === full, "single keeps the full budget");
  assertTrue(paneVoxelBudget(full, "1x2") === 2_000_000, "two panes halve it");
  assertTrue(paneVoxelBudget(full, "2x2") === 1_000_000, "four panes quarter it");
});

check("only active 3D panes divide a 3D voxel budget", () => {
  assertTrue(activeThreeDPaneBudgetScale([false, true]) === 1,
    "a 2D source plus one handed-off volume keeps the full 3D budget");
  assertTrue(activeThreeDPaneBudgetScale([true, true, false]) === 0.5,
    "two active 3D panes share the budget even beside a 2D pane");
  assertTrue(activeThreeDPaneBudgetScale([false, false]) === 1,
    "a layout with no 3D panes retains a neutral scale");
});

/*
 * The point of dividing: total resident volume across panes must not grow with
 * the pane count, because that is what exhausts VRAM and starts costing
 * contexts. Without this, a 2x2 grid would hold 4x the voxels of a single view.
 */
check("total budget across panes never exceeds the single-pane budget", () => {
  const full = 4_000_000;
  for (const layout of PANE_LAYOUTS) {
    const total = paneVoxelBudget(full, layout) * paneCount(layout);
    assertTrue(total <= full, `${layout} total ${total} <= ${full}`);
  }
});

check("the budget never collapses below a usable floor", () => {
  assertTrue(
    paneVoxelBudget(100, "2x2") === MIN_PANE_VOXEL_BUDGET,
    "a tiny budget floors rather than shrinking to nothing",
  );
  assertTrue(paneVoxelBudget(0, "single") === MIN_PANE_VOXEL_BUDGET, "zero floors");
  assertTrue(paneVoxelBudget(Number.NaN, "1x2") === MIN_PANE_VOXEL_BUDGET, "NaN floors");
  assertTrue(paneVoxelBudget(-5, "1x2") === MIN_PANE_VOXEL_BUDGET, "negative floors");
});

// ---------- Camera sync bus ----------

function state(x: number): CameraState {
  return {
    position: [x, 0, 0],
    focalPoint: [0, 0, 0],
    viewUp: [0, 0, 1],
    parallelProjection: false,
    viewAngle: 30,
  };
}

check("a publish reaches every pane except the publisher", () => {
  const bus = createCameraSyncBus();
  const seen: string[] = [];
  bus.subscribe("a", () => seen.push("a"));
  bus.subscribe("b", () => seen.push("b"));
  bus.subscribe("c", () => seen.push("c"));

  bus.publish("a", state(1));
  assertTrue(!seen.includes("a"), "the publisher does not receive its own state");
  assertTrue(seen.includes("b") && seen.includes("c"), "every other pane receives it");
  assertTrue(seen.length === 2, "each other pane receives it exactly once");
});

/*
 * The loop this prevents: applying a camera to a subscriber fires that pane's
 * own modified observer, which republishes. Unguarded, two panes ping-pong
 * until the stack gives out — so a subscriber that republishes on receipt must
 * not be able to re-enter the broadcast.
 */
check("a subscriber republishing on receipt cannot loop", () => {
  const bus = createCameraSyncBus();
  let deliveries = 0;
  bus.subscribe("a", () => {
    deliveries++;
    bus.publish("a", state(99));
  });
  bus.subscribe("b", () => {
    deliveries++;
    bus.publish("b", state(98));
  });

  bus.publish("a", state(1));
  assertTrue(deliveries === 1, `one delivery, got ${deliveries}`);
});

check("republishing an unchanged state is suppressed", () => {
  const bus = createCameraSyncBus();
  let deliveries = 0;
  bus.subscribe("b", () => deliveries++);

  bus.publish("a", state(1));
  bus.publish("a", state(1));
  bus.publish("a", state(1));
  assertTrue(deliveries === 1, `identical states deliver once, got ${deliveries}`);

  bus.publish("a", state(2));
  assertTrue(deliveries === 2, "a genuinely new state still delivers");
});

check("a follower that mounts late replays the authoritative camera", () => {
  const bus = createCameraSyncBus();
  bus.seed("authority", state(7));
  let replay: CameraState | null = null;
  let source = "";
  bus.subscribe("late", (next, from) => {
    replay = next;
    source = from;
  }, { replayLatest: true });
  assertTrue(cameraStatesEqual(replay, state(7)), "late follower receives the seeded camera");
  assertTrue(source === "camera-authority", "replay identifies the camera authority");
});

check("seeding a re-lock replaces the previous shared camera", () => {
  const bus = createCameraSyncBus();
  let lastX = -1;
  bus.subscribe("follower", (next) => { lastX = next.position[0]; });
  bus.publish("old", state(1));
  bus.seed("new-authority", state(4));
  assertTrue(lastX === 4, "the new authority is delivered even after an older shared state");
});

/*
 * A pane mid-teardown can throw on apply. If that escaped the broadcast loop,
 * every pane after it in iteration order would silently stop tracking the
 * camera — sync that half-works is worse than sync that visibly fails.
 * "bad" is registered FIRST so the assertion only passes if delivery continued
 * past the throw.
 */
check("one throwing subscriber does not desynchronise the others", () => {
  const bus = createCameraSyncBus();
  bus.subscribe("bad", () => {
    throw new Error("subscriber exploded");
  });
  let good = 0;
  bus.subscribe("good", () => good++);

  bus.publish("a", state(1));
  assertTrue(good === 1, "the later subscriber still received the state");
  assertTrue(!bus.isApplying(), "the applying flag is cleared");

  bus.publish("a", state(2));
  assertTrue(good === 2, "subsequent publishes keep working");
});

check("unsubscribing stops delivery", () => {
  const bus = createCameraSyncBus();
  let hits = 0;
  const off = bus.subscribe("b", () => hits++);
  bus.publish("a", state(1));
  off();
  bus.publish("a", state(2));
  assertTrue(hits === 1, `delivery stops after unsubscribe, got ${hits}`);
});

/*
 * A remount registers the replacement listener before the old effect's cleanup
 * runs. An unguarded delete in that cleanup would unsubscribe the live pane and
 * silently stop syncing it.
 */
check("a stale unsubscribe cannot remove the live listener", () => {
  const bus = createCameraSyncBus();
  let live = 0;
  const offStale = bus.subscribe("pane", () => {
    throw new Error("stale listener should have been replaced");
  });
  bus.subscribe("pane", () => live++);
  offStale();

  bus.publish("other", state(1));
  assertTrue(live === 1, "the replacement listener is still subscribed");
});

check("cameraStatesEqual compares every component", () => {
  const base = state(1);
  assertTrue(cameraStatesEqual(base, state(1)), "identical states are equal");
  assertTrue(!cameraStatesEqual(base, state(2)), "a moved position differs");
  assertTrue(!cameraStatesEqual(base, null), "null differs from a state");
  assertTrue(cameraStatesEqual(null, null), "two nulls are equal");
  assertTrue(
    !cameraStatesEqual(base, { ...base, viewUp: [0, 1, 0] }),
    "a rolled view-up differs",
  );
  assertTrue(
    !cameraStatesEqual(base, { ...base, parallelScaleM: 2 }),
    "a changed parallel scale differs",
  );
  assertTrue(
    !cameraStatesEqual(base, { ...base, parallelProjection: true }),
    "a changed projection differs",
  );
  assertTrue(
    !cameraStatesEqual(base, { ...base, viewAngle: 35 }),
    "a changed perspective angle differs",
  );
});

// ---------- Summary ----------

for (const line of failures) console.error(line);
console.log(`\npane-layout tests: ${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
