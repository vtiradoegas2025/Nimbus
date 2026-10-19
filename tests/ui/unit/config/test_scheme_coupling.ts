/*
 * Tests for the geometry -> dynamics-scheme coupling in src/lib/configSchema.ts
 * (deriveSchemeFromGeometry + applyFieldChange). Nimbus is Cartesian-only: the
 * coordinate is pinned to cartesian and axisymmetric to false (the Cartesian
 * box has no rotation axis); only the staggering is the user's choice, and the
 * dynamics scheme is DERIVED (read-only) from it. The guarantees under test:
 *   1. each staggering maps to a canonical scheme the C++ loader accepts;
 *   2. selecting a staggering re-pins coordinate=cartesian + axisymmetric=false;
 *   3. changing the staggering re-derives the scheme; other fields don't.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types scripts/test_scheme_coupling.ts
 */

import {
  applyFieldChange,
  defaultsForSchema,
  deriveSchemeFromGeometry,
  readPath,
  reconcileFormConfig,
} from "../../../../ui/src/lib/config/configSchema.ts";
import {
  mergeConfig,
  parseYaml,
  stringifyYaml,
} from "../../../../ui/src/lib/config/yaml.ts";

let passed = 0;
let failed = 0;
const failures: string[] = [];

function assertEq<T>(actual: T, expected: T, label: string): void {
  if (JSON.stringify(actual) === JSON.stringify(expected)) {
    passed++;
  } else {
    failed++;
    failures.push(
      `FAIL ${label}\n        expected: ${JSON.stringify(expected)}\n        actual:   ${JSON.stringify(actual)}`,
    );
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

// ---------- deriveSchemeFromGeometry: staggering -> canonical id ----------

check("each staggering maps to a canonical Cartesian scheme", () => {
  assertEq(
    deriveSchemeFromGeometry("c_grid"),
    "klemp_wilhelmson_split_cartesian_cgrid",
    "c_grid -> cartesian_cgrid",
  );
  assertEq(
    deriveSchemeFromGeometry("collocated"),
    "klemp_wilhelmson_split_cartesian_agrid",
    "collocated -> cartesian_agrid",
  );
});

// ---------- applyFieldChange: geometry coupling + Cartesian constraint ------

check("selecting collocated staggering re-derives the scheme + re-pins geometry", () => {
  const base = defaultsForSchema(); // cartesian, c_grid, axisym false
  const next = applyFieldChange(base, "grid.staggering", "collocated");
  assertEq(readPath(next, "grid.coordinate_system"), "cartesian", "coordinate pinned cartesian");
  assertEq(readPath(next, "dynamics.axisymmetric"), false, "axisym pinned false");
  assertEq(
    readPath(next, "dynamics.scheme"),
    "klemp_wilhelmson_split_cartesian_agrid",
    "scheme derived (collocated -> agrid)",
  );
});

check("switching staggering back to C-grid re-derives cartesian_cgrid", () => {
  const base = defaultsForSchema();
  const collocated = applyFieldChange(base, "grid.staggering", "collocated");
  const back = applyFieldChange(collocated, "grid.staggering", "c_grid");
  assertEq(
    readPath(back, "dynamics.scheme"),
    "klemp_wilhelmson_split_cartesian_cgrid",
    "c_grid -> cartesian_cgrid",
  );
});

check("default config is Cartesian C-grid, non-axisymmetric", () => {
  const base = defaultsForSchema();
  assertEq(readPath(base, "grid.coordinate_system"), "cartesian", "coordinate");
  assertEq(readPath(base, "grid.staggering"), "c_grid", "staggering");
  assertEq(readPath(base, "dynamics.axisymmetric"), false, "axisym");
  assertEq(
    readPath(base, "dynamics.scheme"),
    "klemp_wilhelmson_split_cartesian_cgrid",
    "scheme",
  );
});

check("a non-geometry field writes through without touching the scheme", () => {
  const base = defaultsForSchema();
  const schemeBefore = readPath(base, "dynamics.scheme");
  const next = applyFieldChange(base, "grid.nx", 128);
  assertEq(readPath(next, "grid.nx"), 128, "nx written");
  assertEq(readPath(next, "dynamics.scheme"), schemeBefore, "scheme untouched");
});

check("enabling longwave-capable radiation materializes an explicit skin", () => {
  const base = defaultsForSchema();
  assertEq(readPath(base, "surface.tsk_k"), -1, "PBL default retains initial-state skin");
  const enabled = applyFieldChange(base, "radiation.scheme", "simple_grey");
  assertEq(readPath(enabled, "surface.tsk_k"), 300, "radiation gets explicit 300 K skin");
  assertEq(readPath(base, "surface.tsk_k"), -1, "input surface semantics remain unchanged");
});

check("applyFieldChange does not mutate the input config", () => {
  const base = defaultsForSchema();
  const staggeringBefore = readPath(base, "grid.staggering");
  applyFieldChange(base, "grid.staggering", "collocated");
  assertEq(readPath(base, "grid.staggering"), staggeringBefore, "input unchanged");
});

// ---------- sounding source -> source-specific key coupling ----------------

check("default sounding contains only procedural-CAPE controls", () => {
  const base = defaultsForSchema();
  assertEq(
    readPath(base, "initial_conditions.sounding.type"),
    "parametric_cape",
    "default sounding source",
  );
  assertEq(
    readPath(base, "initial_conditions.sounding.tropopause_m"),
    12000,
    "procedural tropopause present",
  );
  for (const key of ["cin_jkg", "lcl_m", "lfc_m", "el_m", "file_path"]) {
    assertEq(
      readPath(base, `initial_conditions.sounding.${key}`),
      undefined,
      `procedural CAPE omits ${key}`,
    );
  }
});

check("selecting parametric targets swaps to the target-only key family", () => {
  const base = defaultsForSchema();
  const next = applyFieldChange(
    base,
    "initial_conditions.sounding.type",
    "parametric_targets",
  );
  for (const key of ["cape_jkg", "cin_jkg", "lcl_m", "lfc_m", "el_m", "surface_theta_k"]) {
    const present = readPath(next, `initial_conditions.sounding.${key}`) !== undefined;
    assertEq(present, true, `target source contains ${key}`);
  }
  for (const key of ["tropopause_m", "moisture_qv_kgkg", "file_path"]) {
    assertEq(
      readPath(next, `initial_conditions.sounding.${key}`),
      undefined,
      `target source omits ${key}`,
    );
  }
});

check("selecting a file sounding removes every parametric control", () => {
  const targets = applyFieldChange(
    defaultsForSchema(),
    "initial_conditions.sounding.type",
    "parametric_targets",
  );
  const file = applyFieldChange(
    targets,
    "initial_conditions.sounding.type",
    "file",
  );
  assertEq(
    readPath(file, "initial_conditions.sounding.file_path"),
    "",
    "file path is the only source-specific default",
  );
  for (const key of [
    "cape_jkg",
    "cin_jkg",
    "lcl_m",
    "lfc_m",
    "el_m",
    "tropopause_m",
    "surface_theta_k",
    "moisture_qv_kgkg",
  ]) {
    assertEq(
      readPath(file, `initial_conditions.sounding.${key}`),
      undefined,
      `file source omits ${key}`,
    );
  }
});

// ---------- execution policy -> validation coupling -----------------------

check("checkpoint lifecycle is present in structured Studio configuration", () => {
  const base = defaultsForSchema();
  assertEq(readPath(base, "extensions.checkpoint.every_s"), 0, "cadence default");
  assertEq(readPath(base, "extensions.checkpoint.directory"), "", "directory default");
  assertEq(readPath(base, "extensions.checkpoint.resume_from"), "", "resume default");
});

check("execution policy defaults to disclosed exploratory recovery", () => {
  const base = defaultsForSchema();
  assertEq(readPath(base, "extensions.execution.mode"), "exploratory", "mode");
  assertEq(readPath(base, "extensions.execution.acknowledge_maturity_limits"), undefined, "no maturity gate");
  assertEq(readPath(base, "extensions.validation.guard_mode"), "warn", "guard");
});

check("qualified execution pins fail-closed validation", () => {
  const base = defaultsForSchema();
  const qualified = applyFieldChange(base, "extensions.execution.mode", "qualified");
  assertEq(readPath(qualified, "extensions.execution.mode"), "qualified", "mode");
  assertEq(readPath(qualified, "extensions.validation.guard_mode"), "error", "guard");
  assertEq(readPath(qualified, "extensions.execution.acknowledge_maturity_limits"), undefined, "no maturity gate");
  assertEq(readPath(base, "extensions.execution.mode"), "exploratory", "input unchanged");
});

check("uploaded qualified YAML cannot inherit exploratory fallback", () => {
  const uploaded = reconcileFormConfig(mergeConfig(defaultsForSchema(), {
    extensions: { execution: { mode: "qualified" } },
  }));
  assertEq(
    readPath(uploaded, "extensions.compute.allow_fallback"),
    false,
    "qualified upload fallback",
  );
  assertEq(
    readPath(uploaded, "extensions.validation.guard_mode"),
    "error",
    "qualified upload guard",
  );
});

check("qualified scheme changes do not introduce a maturity gate", () => {
  let config = applyFieldChange(defaultsForSchema(), "extensions.execution.mode", "qualified");
  const changed = applyFieldChange(config, "microphysics.scheme", "lin");
  assertEq(readPath(changed, "extensions.execution.acknowledge_maturity_limits"), undefined, "no maturity gate");
});

check("cloud-water bubble seed survives the structured Studio YAML path", () => {
  let config = applyFieldChange(
    defaultsForSchema(),
    "initial_conditions.trigger.type",
    "bubble",
  );
  config = applyFieldChange(
    config,
    "initial_conditions.trigger.bubble.cloud_water_kgkg",
    0.002,
  );
  const restored = parseYaml(stringifyYaml(config));
  assertEq(
    readPath(restored, "initial_conditions.trigger.bubble.cloud_water_kgkg"),
    0.002,
    "cloud-water seed round-trip",
  );
});

check("returning to exploratory restores the disclosed warning policy", () => {
  const qualified = applyFieldChange(
    defaultsForSchema(),
    "extensions.execution.mode",
    "qualified",
  );
  const exploratory = applyFieldChange(
    qualified,
    "extensions.execution.mode",
    "exploratory",
  );
  assertEq(readPath(exploratory, "extensions.validation.guard_mode"), "warn", "guard");
});

console.log(`\nscheme-coupling tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  " + f);
  process.exit(1);
}
process.exit(0);
