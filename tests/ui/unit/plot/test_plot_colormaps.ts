/*
 * Tests for `isDbzField` in src/lib/plotColormaps.ts — the predicate that
 * decides whether a field is painted with the NWS dBZ palette + [-30, 75]
 * range. The key correctness property: the LINEAR `radar` field (mm^6/m^3) is
 * NOT dBZ and must never route to the dBZ scale, while the true dBZ fields
 * (reflectivity_dbz, composite_reflectivity) must.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types scripts/test_plot_colormaps.ts
 */

import {
  isDbzField,
  padDegenerateRange,
  plotlyColorRange,
  uniformRangeValue,
} from "../../../../ui/src/lib/plot/plotColormaps.ts";
import type { FieldUnits } from "../../../../ui/src/lib/science/fieldUnits.ts";

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

const UNITS: FieldUnits = {
  reflectivity_dbz: "dBZ",
  composite_reflectivity: "dBZ",
  radar: "mm^6/m^3",
  w: "m/s",
};

// ---------- Unit is authoritative when present ----------

assertEq(isDbzField("reflectivity_dbz", UNITS), true, "reflectivity_dbz unit=dBZ -> dBZ");
assertEq(isDbzField("composite_reflectivity", UNITS), true, "composite_reflectivity unit=dBZ -> dBZ");
assertEq(isDbzField("radar", UNITS), false, "radar unit=mm^6/m^3 -> NOT dBZ (the core fix)");
assertEq(isDbzField("w", UNITS), false, "w unit=m/s -> NOT dBZ");

// ---------- Name fallback when units are absent (older manifests) ----------

assertEq(isDbzField("reflectivity_dbz", {}), true, "no units: name 'dbz' -> dBZ");
assertEq(isDbzField("composite_reflectivity", null), true, "no units: name 'reflectiv' -> dBZ");
assertEq(isDbzField("radar", {}), false, "no units: linear radar excluded by name");
assertEq(isDbzField("radar", undefined), false, "no units: radar excluded (undefined catalog)");
assertEq(isDbzField("w", {}), false, "no units: plain field -> NOT dBZ");

// ---------- Guards ----------

assertEq(isDbzField(null, UNITS), false, "null field -> NOT dBZ");

// ---------- Range semantics ----------

assertEq(
  JSON.stringify(plotlyColorRange("dBZ", { min: 5, max: 45 })),
  JSON.stringify({ zmin: -30, zmax: 75 }),
  "dBZ anchors do not move with the displayed data",
);
assertEq(
  JSON.stringify(plotlyColorRange("RdBu", { min: -2, max: 7 })),
  JSON.stringify({ zmin: -7, zmax: 7 }),
  "diverging scale is symmetric around zero",
);
assertEq(
  JSON.stringify(plotlyColorRange("ColdPool", { min: 0, max: 0 })),
  JSON.stringify({ zmin: -0.5, zmax: 0.5 }),
  "flat diverging data retains a non-degenerate zero-centred range",
);
assertEq(
  JSON.stringify(plotlyColorRange("Viridis", { min: -2, max: 7 })),
  JSON.stringify({}),
  "sequential scale keeps Plotly's fitted range",
);
assertEq(
  JSON.stringify(plotlyColorRange("Viridis", { min: 0, max: 0 })),
  JSON.stringify({ zmin: -0.5, zmax: 0.5 }),
  "flat sequential data gets an explicit non-degenerate range",
);
assertEq(uniformRangeValue({ min: 3, max: 3 }), 3, "uniform range exposes its constant");
assertEq(uniformRangeValue({ min: 3, max: 4 }), null, "varying range is not uniform");
assertEq(
  JSON.stringify(padDegenerateRange({ min: 20_000, max: 20_000 })),
  JSON.stringify({ min: 19_980, max: 20_020 }),
  "large constants use relative padding",
);

// ---------- Report ----------

console.log(`\nplotColormaps (isDbzField) tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  " + f);
  process.exit(1);
}
process.exit(0);
