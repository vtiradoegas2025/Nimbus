/*
 * Regression tests for `src/lib/diagnostics.ts` v1 config-path reads.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types scripts/test_diagnostics.ts
 *
 * Bugs under test:
 *   (a) CFL + step count read a vestigial `grid.dt` instead of the canonical
 *       `time.dt_large`, so editing the real timestep didn't move the readout
 *       (masked because both default to 0.5). Run length read a dropped
 *       top-level `duration_s` instead of `time.run_time`.
 *   (b) The scheme/coordinate compatibility warning: Nimbus is Cartesian-only,
 *       so every recognized scheme requires coordinate_system: cartesian. The
 *       check mainly guards an uploaded/hand-edited config that still carries a
 *       stale (e.g. legacy "cylindrical") coordinate value.
 */

import { computeDiagnostics } from "../../../../ui/src/lib/science/diagnostics.ts";
import type { ConfigObject } from "../../../../ui/src/lib/config/configSchema.ts";

// ---------- Tiny assertion API (matches test_schema_validator.ts) ----------

let failed = 0;
let passed = 0;
const failures: string[] = [];

function assertEq<T>(actual: T, expected: T, label: string): void {
  if (actual === expected) {
    passed++;
  } else {
    failed++;
    failures.push(
      `FAIL ${label}\n        expected: ${JSON.stringify(expected)}\n        actual:   ${JSON.stringify(actual)}`,
    );
  }
}

function assertTrue(cond: boolean, label: string): void {
  assertEq(cond, true, label);
}

function check(label: string, fn: () => void): void {
  try {
    fn();
  } catch (e) {
    failed++;
    failures.push(`THREW ${label}: ${(e as Error).message}`);
  }
}

// ---------- Helpers ----------

function statValue(config: ConfigObject, label: string): string | undefined {
  return computeDiagnostics(config).stats.find((s) => s.label === label)?.value;
}

function hasCoordWarning(config: ConfigObject): boolean {
  return computeDiagnostics(config).warnings.some((w) =>
    w.message.includes("coordinate_system"),
  );
}

function hasDrySoundingWarning(config: ConfigObject): boolean {
  return computeDiagnostics(config).warnings.some((w) =>
    w.message.includes("no moisture available"),
  );
}

// ---------- (a) timestep + run length read canonical v1 paths ----------

check("CFL reads time.dt_large, not the vestigial grid.dt", () => {
  // dx=500, dt_large=5 -> CFL = 5*50/500 = 0.50. grid.dt is set to a wildly
  // different value to prove it is ignored.
  const config = {
    grid: { nx: 32, ny: 32, nz: 32, dx: 500, dy: 500, dz: 500, dt: 0.01, coordinate_system: "cartesian" },
    time: { dt_large: 5, run_time: 1000 },
    dynamics: { scheme: "klemp_wilhelmson_split_cartesian_cgrid" },
  } as unknown as ConfigObject;
  assertEq(statValue(config, "CFL"), "0.50", "CFL derived from time.dt_large");
});

check("steps read time.run_time / time.dt_large", () => {
  // run_time=1000, dt_large=5 -> 200 steps.
  const config = {
    grid: { nx: 32, ny: 32, nz: 32, dx: 500, dy: 500, dz: 500, coordinate_system: "cartesian" },
    time: { dt_large: 5, run_time: 1000 },
    dynamics: { scheme: "klemp_wilhelmson_split_cartesian_cgrid" },
  } as unknown as ConfigObject;
  assertEq(statValue(config, "steps"), "200", "steps from run_time/dt_large");
});

check("editing time.dt_large moves the CFL readout", () => {
  const base = {
    grid: { nx: 32, ny: 32, nz: 32, dx: 500, dy: 500, dz: 500, coordinate_system: "cartesian" },
    dynamics: { scheme: "klemp_wilhelmson_split_cartesian_cgrid" },
  };
  const slow = { ...base, time: { dt_large: 1, run_time: 1000 } } as unknown as ConfigObject;
  const fast = { ...base, time: { dt_large: 10, run_time: 1000 } } as unknown as ConfigObject;
  assertTrue(
    statValue(slow, "CFL") !== statValue(fast, "CFL"),
    "CFL changes when dt_large changes",
  );
});

// ---------- (b) scheme/coordinate compatibility (Cartesian-only) ----------

check("canonical cartesian scheme on a stale cylindrical-typed grid warns", () => {
  const config = {
    grid: { nx: 32, ny: 32, nz: 32, dx: 500, dy: 500, dz: 500, coordinate_system: "cylindrical" },
    time: { dt_large: 0.5, run_time: 1000 },
    dynamics: { scheme: "klemp_wilhelmson_split_cartesian_cgrid" },
  } as unknown as ConfigObject;
  assertTrue(hasCoordWarning(config), "warns: Cartesian scheme needs coordinate_system cartesian");
});

check("cartesian scheme on a cartesian grid produces no compatibility warning", () => {
  const config = {
    grid: { nx: 32, ny: 32, nz: 32, dx: 500, dy: 500, dz: 500, coordinate_system: "cartesian" },
    time: { dt_large: 0.5, run_time: 1000 },
    dynamics: { scheme: "klemp_wilhelmson_split_cartesian_cgrid" },
  } as unknown as ConfigObject;
  assertTrue(!hasCoordWarning(config), "no warning when coords match the scheme");
});

check("legacy cartesian alias still triggers the compatibility check", () => {
  const config = {
    grid: { nx: 32, ny: 32, nz: 32, dx: 500, dy: 500, dz: 500, coordinate_system: "cylindrical" },
    time: { dt_large: 0.5, run_time: 1000 },
    dynamics: { scheme: "cart" },
  } as unknown as ConfigObject;
  assertTrue(hasCoordWarning(config), "legacy alias 'cart' on a stale coord still warns");
});

check("unknown scheme stays quiet (no spurious warning)", () => {
  const config = {
    grid: { nx: 32, ny: 32, nz: 32, dx: 500, dy: 500, dz: 500, coordinate_system: "cartesian" },
    time: { dt_large: 0.5, run_time: 1000 },
    dynamics: { scheme: "" },
  } as unknown as ConfigObject;
  assertTrue(!hasCoordWarning(config), "empty/unknown scheme does not warn");
});

check("target-derived and file sounding moisture are not mistaken for zero", () => {
  const base = {
    grid: { nx: 32, ny: 32, nz: 32, dx: 500, dy: 500, dz: 500, coordinate_system: "cartesian" },
    time: { dt_large: 0.5, run_time: 1000 },
    dynamics: { scheme: "klemp_wilhelmson_split_cartesian_cgrid" },
    microphysics: { scheme: "kessler" },
  };
  const targets = {
    ...base,
    initial_conditions: { sounding: { type: "parametric_targets", lcl_m: 1200 } },
  } as unknown as ConfigObject;
  const file = {
    ...base,
    initial_conditions: { sounding: { type: "file", file_path: "sounding.txt" } },
  } as unknown as ConfigObject;
  assertTrue(!hasDrySoundingWarning(targets), "targets derives qv from LCL");
  assertTrue(!hasDrySoundingWarning(file), "file obtains qv from its column");
});

// ---------- Report ----------

console.log(`\ndiagnostics tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  " + f);
  process.exit(1);
}
process.exit(0);
