/*
 * Tests for src/lib/vectorComponents.ts — the plane-aware default component
 * picker for the VectorField (quiver) plot.
 *
 * Regression under test: VectorField used to default to (w, u) regardless of
 * the slice plane. In a plan-view category (z-slice) that drew vertical
 * velocity w as a horizontal arrow component and omitted the y-wind v.
 * defaultComponents must pick the IN-PLANE components for each slice axis.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types scripts/test_vector_components.ts
 */

import {
  availableVectorPairs,
  defaultComponents,
  inPlaneAxes,
} from "../../../../ui/src/lib/science/vectorComponents.ts";
import type { SliceAxis } from "../../../../ui/src/lib/platform/npy.ts";

// ---------- Tiny assertion API (matches test_quiver.ts) ----------

let failed = 0;
let passed = 0;
const failures: string[] = [];

function assertEq(actual: unknown, expected: unknown, label: string): void {
  if (actual === expected) passed++;
  else {
    failed++;
    failures.push(`FAIL ${label}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

const UVW = ["u", "v", "w"];

// ---------- inPlaneAxes: meteorological display orientation ----------

assertEq(JSON.stringify(inPlaneAxes(2 as SliceAxis)), JSON.stringify({ rowAxis: 0, colAxis: 1 }), "z-slice plane = (x, y)");
assertEq(JSON.stringify(inPlaneAxes(1 as SliceAxis)), JSON.stringify({ rowAxis: 2, colAxis: 0 }), "y-slice displays x rightward and z upward");
assertEq(JSON.stringify(inPlaneAxes(0 as SliceAxis)), JSON.stringify({ rowAxis: 2, colAxis: 1 }), "x-slice displays y rightward and z upward");

// ---------- Plane-aware defaults (the core fix) ----------

// z-slice = plan view: horizontal wind (x-arrow = y comp = v, y-arrow = x comp = u). NO w.
const plan = defaultComponents(2 as SliceAxis, UVW);
assertEq(plan.xField, "v", "plan view xField = v (y component)");
assertEq(plan.yField, "u", "plan view yField = u (x component)");
assertEq(plan.xField !== "w" && plan.yField !== "w", true, "plan view never defaults to vertical velocity w");

// y-slice = x-z plane: horizontal velocity rightward, vertical velocity upward.
const ySlice = defaultComponents(1 as SliceAxis, UVW);
assertEq(ySlice.xField, "u", "y-slice xField = u (x component)");
assertEq(ySlice.yField, "w", "y-slice yField = w (z component)");

// x-slice = y-z plane: horizontal velocity rightward, vertical velocity upward.
const xSlice = defaultComponents(0 as SliceAxis, UVW);
assertEq(xSlice.xField, "v", "x-slice xField = v (y component)");
assertEq(xSlice.yField, "w", "x-slice yField = w (z component)");

// ---------- Scalars are never silently treated as vector components ----------

const fb = defaultComponents(2 as SliceAxis, ["temperature", "pressure"]);
assertEq(fb.xField, null, "scalar-only run has no x vector component");
assertEq(fb.yField, null, "scalar-only run has no y vector component");

const vort = defaultComponents(
  2 as SliceAxis,
  ["vorticity_r", "vorticity_theta", "vorticity_z"],
);
assertEq(vort.xField, "vorticity_theta", "plan vorticity x arrow uses y component");
assertEq(vort.yField, "vorticity_r", "plan vorticity y arrow uses x component");

const families = availableVectorPairs(
  2 as SliceAxis,
  [...UVW, "vorticity_r", "vorticity_theta", "vorticity_z"],
);
assertEq(families.length, 2, "velocity and vorticity are separate valid families");
assertEq(families[0].family.id, "velocity", "velocity remains the default family");

assertEq(defaultComponents(2 as SliceAxis, []).xField, null, "empty fields -> xField null");
assertEq(defaultComponents(2 as SliceAxis, []).yField, null, "empty fields -> yField null");

// ---------- Report ----------

console.log(`\nvectorComponents tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  " + f);
  process.exit(1);
}
process.exit(0);
