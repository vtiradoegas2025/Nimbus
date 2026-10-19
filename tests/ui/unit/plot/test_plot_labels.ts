/*
 * Tests for src/lib/plot/plotLabels.ts.
 *
 * Two label faults motivated this module:
 *   1. EnsembleAnalysisPlot labelled its colorbars "Mean" and "Spread" and its
 *      annotation "Probability > 30" with NO unit anywhere, even though the
 *      reference step carries fieldUnits and eligibleFields is already filtered
 *      for unit consistency. The identical label meant dBZ for one field and
 *      m/s for another, which in a science UI is the worst class of fault.
 *   2. CompositePlot reported a contour layer skipped for a PLANE mismatch but
 *      stayed silent when the layer had no levels -- the case the default
 *      picker reaches on its own, since it enables any field it matches while
 *      `defaultLevels` only has a default for dBZ and vorticity fields. An
 *      enabled, populated layer drew nothing with no explanation.
 *
 * Run via:
 *   cd ui && npm run test:plot-labels
 */

import {
  describeContourSkips,
  statisticLabel,
  thresholdLabel,
  uniformFieldLabel,
  scalarColorRangeLabel,
  type ContourSkip,
} from "../../../../ui/src/lib/plot/plotLabels.ts";

let passed = 0;
let failed = 0;
const failures: string[] = [];

function check(label: string, condition: boolean): void {
  if (condition) {
    passed++;
  } else {
    failed++;
    failures.push(label);
  }
}

const UNITS = { reflectivity_dbz: "dBZ", w: "m s-1", okubo_weiss: "1", cloud: "kg kg-1" };

/* ---------- Statistic labels carry the field's unit ---------- */

check(
  "a statistic label carries the unit",
  statisticLabel("Mean", "reflectivity_dbz", UNITS) === "Mean (dBZ)",
);
check(
  "the same statistic differs by field",
  statisticLabel("Mean", "w", UNITS) === "Mean (m s-1)",
);
check(
  "spread uses the field's unit too, not a fixed one",
  statisticLabel("Spread", "cloud", UNITS) === "Spread (kg kg-1)",
);
// The regression is precisely that two different fields produced the SAME label.
check(
  "two fields never share a label",
  statisticLabel("Mean", "reflectivity_dbz", UNITS) !== statisticLabel("Mean", "w", UNITS),
);

/* ---------- Dimensionless and unknown units stay bare ---------- */

// "1" is the CF convention for dimensionless; rendering "(1)" would be noise.
check(
  "a dimensionless field gets no parenthetical",
  statisticLabel("Mean", "okubo_weiss", UNITS) === "Mean",
);
check(
  "an unknown field degrades to the bare statistic",
  statisticLabel("Mean", "not_in_the_table", UNITS) === "Mean",
);
check("an empty unit table degrades safely", statisticLabel("Mean", "w", {}) === "Mean");

/* ---------- Threshold labels put the unit on the NUMBER ---------- */

check(
  "the threshold unit trails the value",
  thresholdLabel(30, "reflectivity_dbz", UNITS) === "Probability > 30 dBZ",
);
check(
  "a dimensionless threshold stays bare",
  thresholdLabel(0.01, "okubo_weiss", UNITS) === "Probability > 0.01",
);
check(
  "a negative threshold is rendered verbatim",
  thresholdLabel(-5, "w", UNITS) === "Probability > -5 m s-1",
);

/* ---------- Uniform fields disclose their exact value ---------- */

check(
  "a zero field is identified rather than looking failed",
  uniformFieldLabel({ min: 0, max: 0 }, "stp") === "Uniform field · stp = 0",
);
check(
  "a uniform physical field keeps its unit-bearing label",
  uniformFieldLabel({ min: 2.5, max: 2.5 }, "w (m s-1)")
    === "Uniform field · w (m s-1) = 2.5",
);
check(
  "a varying field receives no uniform disclosure",
  uniformFieldLabel({ min: 0, max: 1 }, "stp") === null,
);

/* ---------- Contour skips: both causes are reported, separately ---------- */

const PLANE = "x–y";

check("nothing skipped produces no lines", describeContourSkips([], PLANE).length === 0);

const planeOnly: ContourSkip[] = [{ field: "composite_reflectivity", reason: "plane" }];
const planeLines = describeContourSkips(planeOnly, PLANE);
check("a plane mismatch is reported", planeLines.length === 1);
check("the plane line names the plane", (planeLines[0] ?? "").includes("x–y"));
check("the plane line names the field", (planeLines[0] ?? "").includes("composite_reflectivity"));

// The regression: this case used to produce NO line at all.
const levelsOnly: ContourSkip[] = [{ field: "radar", reason: "levels" }];
const levelLines = describeContourSkips(levelsOnly, PLANE);
check("a level-less layer is reported", levelLines.length === 1);
check("the level line names the field", (levelLines[0] ?? "").includes("radar"));
check("the level line states the remedy", /enter one or more values/.test((levelLines[0] ?? "")));
check(
  "a level-less layer is not blamed on the plane",
  !(levelLines[0] ?? "").includes("plane"),
);

const rangeOnly: ContourSkip[] = [{ field: "vorticity_z", reason: "range" }];
const rangeLines = describeContourSkips(rangeOnly, PLANE);
check("an out-of-range contour is reported", rangeLines.length === 1);
check("the range warning names the field", (rangeLines[0] ?? "").includes("vorticity_z"));
check("the range warning states the remedy", /adjust levels/.test(rangeLines[0] ?? ""));

/* ---------- Mixed causes stay distinguishable ---------- */

const mixed: ContourSkip[] = [
  { field: "composite_reflectivity", reason: "plane" },
  { field: "radar", reason: "levels" },
];
const mixedLines = describeContourSkips(mixed, PLANE);
check("each cause gets its own line", mixedLines.length === 2);
check(
  "the plane line mentions only its own field",
  (mixedLines[0] ?? "").includes("composite_reflectivity") && !(mixedLines[0] ?? "").includes("radar"),
);
check(
  "the level line mentions only its own field",
  (mixedLines[1] ?? "").includes("radar") && !(mixedLines[1] ?? "").includes("composite_reflectivity"),
);

// Both contour layers can fail the same way at once.
const twoLevelless: ContourSkip[] = [
  { field: "radar", reason: "levels" },
  { field: "vorticity_magnitude", reason: "levels" },
];
const grouped = describeContourSkips(twoLevelless, PLANE);
check("same-cause skips share one line", grouped.length === 1);
check(
  "the shared line names every field",
  (grouped[0] ?? "").includes("radar") && (grouped[0] ?? "").includes("vorticity_magnitude"),
);

check("missing values are distinguished from a uniform scalar", scalarColorRangeLabel(null, { min: 0, max: 1 }) === "No finite data in this slice");
check("lower-end clipping is disclosed", scalarColorRangeLabel({ min: -4, max: -2 }, { min: 0, max: 1 })?.includes("below") === true);
check("upper-end clipping is disclosed", scalarColorRangeLabel({ min: 2, max: 4 }, { min: 0, max: 1 })?.includes("above") === true);
check("a narrow nonconstant wind range is disclosed", scalarColorRangeLabel({ min: -3, max: 1 }, { min: -30, max: 75 }) === "Visible range uses 3.8% of the color scale");
check("a well-resolved range adds no notice", scalarColorRangeLabel({ min: -3, max: 1 }, { min: -3, max: 3 }) === null);
check("small nonzero fractions never round to zero", scalarColorRangeLabel({ min: 0, max: 1e-6 }, { min: 0, max: 1 })?.includes("<0.1%") === true);

console.log(`\nplot-label tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  " + f);
  process.exit(1);
}
process.exit(0);
