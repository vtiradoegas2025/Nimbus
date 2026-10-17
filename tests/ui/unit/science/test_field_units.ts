/*
 * Tests for `src/lib/fieldUnits.ts` — the colorbar/axis unit labeller used by
 * every 2D plot. Units flow from the run manifest's field catalog (emitted by
 * the simulator from field_contract.cpp), so these tests pin the UI-side
 * formatting rules: dimensionless "1" and unknown fields get no suffix.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types scripts/test_field_units.ts
 */

import { unitFor, labelWithUnit, type FieldUnits } from "../../../../ui/src/lib/science/fieldUnits.ts";

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
  u: "m/s",
  w: "m/s",
  theta: "K",
  p: "Pa",
  qv: "kg/kg",
  radar: "mm^6/m^3",
  reflectivity_dbz: "dBZ",
  vorticity_z: "1/s",
  tracer: "1", // dimensionless marker
};

// ---------- unitFor ----------

assertEq(unitFor("u", UNITS), "m/s", "unitFor known field");
assertEq(unitFor("reflectivity_dbz", UNITS), "dBZ", "unitFor dBZ field");
assertEq(unitFor("radar", UNITS), "mm^6/m^3", "unitFor linear radar field");
assertEq(unitFor("tracer", UNITS), null, 'unitFor suppresses dimensionless "1"');
assertEq(unitFor("does_not_exist", UNITS), null, "unitFor unknown field -> null");
assertEq(unitFor("u", null), null, "unitFor null units -> null");
assertEq(unitFor("u", undefined), null, "unitFor undefined units -> null");
assertEq(unitFor("u", {}), null, "unitFor empty catalog -> null");
assertEq(unitFor("pbl_height", null), "m", "legacy pbl_height recovers its canonical metre unit");

// ---------- labelWithUnit ----------

assertEq(labelWithUnit("w", UNITS), "w (m/s)", "labelWithUnit appends unit");
assertEq(labelWithUnit("qv", UNITS), "qv (kg/kg)", "labelWithUnit kg/kg");
assertEq(labelWithUnit("vorticity_z", UNITS), "vorticity_z (1/s)", "labelWithUnit 1/s");
assertEq(labelWithUnit("tracer", UNITS), "tracer", "labelWithUnit bare name for dimensionless");
assertEq(labelWithUnit("unknown", UNITS), "unknown", "labelWithUnit bare name for unknown");
assertEq(labelWithUnit("w", null), "w", "labelWithUnit bare name when no catalog");
assertEq(labelWithUnit("pbl_height", null), "pbl_height (m)", "legacy pbl_height colorbar keeps metres");

// ---------- Report ----------

console.log(`\nfieldUnits tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  " + f);
  process.exit(1);
}
process.exit(0);
