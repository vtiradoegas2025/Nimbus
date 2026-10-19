/*
 * Smoke + unit tests for `src/lib/schemaValidator.ts` (CM1Mapping §21.D1).
 *
 * Run via:
 *   cd ui && node --experimental-strip-types scripts/test_schema_validator.ts
 *
 * (Node 22.7+ supports --experimental-strip-types; the import path uses
 * .ts extensions which `allowImportingTsExtensions` already permits.)
 *
 * Coverage:
 *   1. v1_valid: every migrated `tests/fixtures/configs/test/*.yaml` round-trips clean.
 *   2. v1_with_warnings: extra unknown top-level section produces warnings,
 *      not errors.
 *   3. v1_invalid: section value not an object → blocking error.
 *   4. v0_legacy: top-level duration_s / environment → v0_legacy.
 *   5. stamp_missing: object without `schema:` and without v0 tells.
 *   6. stamp_unknown: `schema: nimbus.v2` → stamp_unknown.
 *   7. not_an_object: top-level array → not_an_object.
 *   8. isBlocking: gates v1_valid / v1_with_warnings as non-blocking;
 *      everything else blocks.
 */

import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { parse as parseYaml } from "yaml";

import {
  isBlocking,
  SCHEMA_V1_STAMP,
  summarize,
  V1_TOP_LEVEL_KEYS,
  validateSchemaVersion,
  type SchemaValidationResult,
} from "../../../../ui/src/lib/config/schemaValidator.ts";
import { defaultsForSchema } from "../../../../ui/src/lib/config/configSchema.ts";

// ---------- Tiny assertion API (no test framework) ----------

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

// ---------- Fixtures ----------

const HERE = dirname(fileURLToPath(import.meta.url));
// tests/ui/unit/config -> repository root
const CONFIGS_DIR = join(
  HERE, "..", "..", "..", "..", "tests", "fixtures", "configs", "test",
);

function listMigratedYamls(): string[] {
  // Top-level tests/fixtures/configs/test/*.yaml only; legacy/ subdir holds v0 originals
  // and is deliberately out of scope for v1 validation.
  return readdirSync(CONFIGS_DIR)
    .filter((f) => f.endsWith(".yaml"))
    .map((f) => join(CONFIGS_DIR, f));
}

// ---------- 1. v1_valid: every tests/fixtures/configs/test/*.yaml ----------

check("v1_valid: all migrated tests/fixtures/configs/test/*.yaml", () => {
  const files = listMigratedYamls();
  assertTrue(files.length > 0, "found at least one yaml fixture");
  for (const path of files) {
    const text = readFileSync(path, "utf-8");
    const parsed = parseYaml(text);
    const r = validateSchemaVersion(parsed);
    assertEq(
      r.status,
      "v1_valid",
      `  ${path.split("/").pop()}: expected v1_valid, got ${r.status} (${summarize(r)})`,
    );
    assertEq(r.schemaStamp, SCHEMA_V1_STAMP, `  ${path}: schemaStamp`);
    assertEq(r.errors.length, 0, `  ${path}: errors empty`);
    assertEq(r.warnings.length, 0, `  ${path}: warnings empty`);
    assertEq(isBlocking(r), false, `  ${path}: not blocking`);
  }
});

// ---------- 2. v1_with_warnings: unknown top-level section ----------

check("v1_with_warnings: unknown top-level section warns, doesn't block", () => {
  const r = validateSchemaVersion({
    schema: "nimbus.v1",
    grid: {},
    something_unknown: { whatever: 1 },
  });
  assertEq(r.status, "v1_with_warnings", "status");
  assertEq(r.errors.length, 0, "errors empty");
  assertEq(r.warnings.length, 1, "one warning");
  assertEq(r.warnings[0].code, "unknown_section", "warning code");
  assertEq(r.warnings[0].path, "something_unknown", "warning path");
  assertEq(isBlocking(r), false, "non-blocking");
});

// ---------- 3. v1_invalid: section value not an object ----------

check("v1_invalid: section value not an object blocks", () => {
  const r = validateSchemaVersion({
    schema: "nimbus.v1",
    grid: "this should be an object",
  });
  assertEq(r.status, "v1_invalid", "status");
  assertEq(r.errors.length, 1, "one error");
  assertEq(r.errors[0].code, "section_not_object", "error code");
  assertEq(r.errors[0].path, "grid", "error path");
  assertEq(isBlocking(r), true, "blocking");
});

// ---------- 4. v0_legacy: top-level duration_s / environment ----------

check("v0_legacy: duration_s at root detected as v0", () => {
  const r = validateSchemaVersion({
    duration_s: 1200,
    coordinate_system: "cartesian",
    environment: { sounding: { type: "parametric_cape" } },
  });
  assertEq(r.status, "v0_legacy", "status");
  assertEq(r.schemaStamp, null, "no stamp");
  assertEq(r.errors[0].code, "stamp_v0_legacy", "error code");
  assertEq(isBlocking(r), true, "blocking");
});

// ---------- 5. stamp_missing: empty object ----------

check("stamp_missing: empty object → stamp_missing", () => {
  const r = validateSchemaVersion({});
  assertEq(r.status, "stamp_missing", "status");
  assertEq(r.errors[0].code, "stamp_missing", "error code");
  assertEq(isBlocking(r), true, "blocking");
});

check("stamp_missing: object with only v1 sections but no stamp", () => {
  const r = validateSchemaVersion({ grid: {}, dynamics: {} });
  assertEq(r.status, "stamp_missing", "status");
  assertEq(isBlocking(r), true, "blocking");
});

// ---------- 6. stamp_unknown ----------

check("stamp_unknown: schema: nimbus.v2 → stamp_unknown", () => {
  const r = validateSchemaVersion({ schema: "nimbus.v2", grid: {} });
  assertEq(r.status, "stamp_unknown", "status");
  assertEq(r.schemaStamp, "nimbus.v2", "stamp preserved");
  assertEq(isBlocking(r), true, "blocking");
});

check("stamp_unknown: non-string stamp", () => {
  const r = validateSchemaVersion({ schema: 1, grid: {} });
  assertEq(r.status, "stamp_unknown", "status");
  assertEq(r.errors[0].code, "stamp_not_string", "error code");
  assertEq(isBlocking(r), true, "blocking");
});

// ---------- 7. not_an_object ----------

check("not_an_object: top-level array blocks", () => {
  const r = validateSchemaVersion([1, 2, 3]);
  assertEq(r.status, "not_an_object", "status");
  assertEq(r.errors[0].code, "not_an_object", "error code");
  assertEq(isBlocking(r), true, "blocking");
});

check("not_an_object: top-level scalar blocks", () => {
  const r = validateSchemaVersion(42);
  assertEq(r.status, "not_an_object", "status");
  assertEq(isBlocking(r), true, "blocking");
});

check("not_an_object: null blocks", () => {
  const r = validateSchemaVersion(null);
  assertEq(r.status, "not_an_object", "status");
  assertEq(isBlocking(r), true, "blocking");
});

// ---------- 8. summarize covers every SchemaStatus ----------

check("summarize: returns a non-empty string for every status", () => {
  const samples: SchemaValidationResult[] = [
    { status: "v1_valid", schemaStamp: SCHEMA_V1_STAMP, errors: [], warnings: [] },
    {
      status: "v1_with_warnings",
      schemaStamp: SCHEMA_V1_STAMP,
      errors: [],
      warnings: [{ code: "unknown_section", message: "x" }],
    },
    {
      status: "v1_invalid",
      schemaStamp: SCHEMA_V1_STAMP,
      errors: [{ code: "section_not_object", message: "x" }],
      warnings: [],
    },
    {
      status: "v0_legacy",
      schemaStamp: null,
      errors: [{ code: "stamp_v0_legacy", message: "x" }],
      warnings: [],
    },
    {
      status: "stamp_missing",
      schemaStamp: null,
      errors: [{ code: "stamp_missing", message: "x" }],
      warnings: [],
    },
    {
      status: "stamp_unknown",
      schemaStamp: "weird",
      errors: [{ code: "stamp_unknown", message: "x" }],
      warnings: [],
    },
    {
      status: "not_an_object",
      schemaStamp: null,
      errors: [{ code: "not_an_object", message: "x" }],
      warnings: [],
    },
  ];
  for (const s of samples) {
    const out = summarize(s);
    assertTrue(out.length > 0, `summarize(${s.status}) non-empty`);
  }
});

// ---------- 9. defaultsForSchema round-trip (D2 invariant) ----------
//
// The Studio's `defaultsForSchema()` output is the seed for the "New
// config" flow and the merge target for every uploaded YAML. After D2
// it must include the `schema: nimbus.v1` stamp and validate cleanly.
// Unknown-section warnings are OK (legacy v0 sections like chaos /
// system / numerics still live at root until D2.2 collapses them
// under extensions:).

check("D2: defaultsForSchema() passes v1 validation", () => {
  const defaults = defaultsForSchema();
  const r = validateSchemaVersion(defaults);
  assertEq(r.schemaStamp, SCHEMA_V1_STAMP, "stamp present");
  assertTrue(
    r.status === "v1_valid" || r.status === "v1_with_warnings",
    `defaultsForSchema status: ${r.status} (${summarize(r)})`,
  );
  assertEq(r.errors.length, 0, "no blocking errors");
  assertEq(isBlocking(r), false, "non-blocking");
});

check("D2: defaultsForSchema() contains every new v1 section", () => {
  const defaults = defaultsForSchema();
  for (const key of [
    "time",
    "boundary_conditions",
    "damping",
    "coriolis",
    "surface",
  ]) {
    assertTrue(
      key in defaults,
      `defaultsForSchema() has top-level \`${key}\``,
    );
    assertTrue(
      V1_TOP_LEVEL_KEYS.has(key),
      `validator recognizes \`${key}\` as v1 top-level`,
    );
  }
});

check("D2: defaultsForSchema().time.* sane defaults", () => {
  const defaults = defaultsForSchema();
  const time = defaults.time as Record<string, unknown>;
  assertEq(time.run_time, 1200, "time.run_time = 1200 s");
  assertEq(time.dt_large, 0.5, "time.dt_large = 0.5 s");
  assertEq(time.adaptive_dt, false, "time.adaptive_dt = false");
  assertEq(time.cfl_safety, 0.7, "time.cfl_safety = 0.7");
});

check("D2: defaultsForSchema().damping mirrors cgrid sponge defaults", () => {
  const defaults = defaultsForSchema();
  const damping = defaults.damping as Record<string, unknown>;
  assertEq(damping.enabled, true, "damping.enabled = true");
  assertEq(damping.alpha, 0.05, "damping.alpha = 0.05 (cgrid-tuned)");
  assertEq(damping.mode, "explicit", "damping.mode = explicit");
});

check("D2: defaultsForSchema().coriolis WK 35°N default", () => {
  const defaults = defaultsForSchema();
  const coriolis = defaults.coriolis as Record<string, unknown>;
  assertEq(coriolis.latitude_deg, 35, "coriolis.latitude_deg = 35");
  assertEq(coriolis.mode, "f_plane", "coriolis.mode = f_plane");
});

// ---------- D2.2 — top-level cleanup + renest ----------

check("D2.2: defaults have no top-level v0 tells (duration_s, coordinate_system)", () => {
  const defaults = defaultsForSchema();
  assertTrue(
    !("duration_s" in defaults),
    "duration_s no longer at root (moved to time.run_time)",
  );
  assertTrue(
    !("coordinate_system" in defaults),
    "coordinate_system no longer at root (moved to grid.coordinate_system)",
  );
});

check("D2.2: coordinate_system lives under grid", () => {
  const defaults = defaultsForSchema();
  const grid = defaults.grid as Record<string, unknown>;
  assertEq(grid.coordinate_system, "cartesian", "grid.coordinate_system default");
});

check("D2.2: hodograph + trigger renested under initial_conditions", () => {
  const defaults = defaultsForSchema();
  const ic = defaults.initial_conditions as Record<string, unknown>;
  assertTrue(
    ic !== undefined && typeof ic === "object",
    "initial_conditions section exists",
  );
  assertTrue(
    "hodograph" in ic,
    "initial_conditions.hodograph present (was environment.hodograph)",
  );
  assertTrue(
    "trigger" in ic,
    "initial_conditions.trigger present (was top-level trigger)",
  );
});

check("SCI-094: defaults preserve the strict terrain selection contract", () => {
  const defaults = defaultsForSchema();
  const terrain = defaults.terrain as Record<string, unknown>;
  assertEq(terrain.scheme, "none", "terrain.scheme default");
  assertEq(terrain.coordinate, "btf", "terrain.coordinate default");
  assertEq(terrain.compute_derivatives, true, "terrain derivatives default");
  assertEq(terrain.compute_metrics, true, "terrain metrics default");
  assertTrue(
    typeof terrain.bell === "object" && terrain.bell !== null,
    "terrain bell parameters are represented",
  );
  assertTrue(
    typeof terrain.schar === "object" && terrain.schar !== null,
    "terrain Schär parameters are represented",
  );
});

// ---------- D3 — dynamics scheme defaults are v1 canonical ----------

check("D3: defaultsForSchema().dynamics defaults to canonical compound id", () => {
  const defaults = defaultsForSchema();
  const dynamics = defaults.dynamics as Record<string, unknown>;
  assertEq(
    dynamics.scheme,
    "klemp_wilhelmson_split_cartesian_cgrid",
    "dynamics.scheme is canonical v1 id (Cartesian C-grid)",
  );
  assertEq(
    dynamics.axisymmetric,
    false,
    "dynamics.axisymmetric pinned off (Cartesian box has no rotation axis)",
  );
  assertTrue(
    typeof dynamics.scheme === "string" &&
      (dynamics.scheme as string).startsWith("klemp_wilhelmson_split_"),
    "default scheme prefixed with klemp_wilhelmson_split_",
  );
});

// ---------- 10. D4 migration round-trip ----------
//
// Simulate what D4's `migrate_config` Tauri command does: spawn
// `bin/nimbus_sim --migrate-config PATH` against a real v0 fixture,
// capture stdout, and re-validate it as v1. This is the loop-closing
// smoke for the "Migrate to v1" CTA — if the migrator's output ever
// stops validating, this test catches it.
//
// Skipped automatically when bin/nimbus_sim isn't built (Studio devs
// without the simulator binary in their tree).

const PROJECT_ROOT = join(HERE, "..", "..", "..");
const SIM_BINARY = join(PROJECT_ROOT, "bin", "nimbus_sim");
const LEGACY_DIR = join(
  PROJECT_ROOT, "tests", "fixtures", "configs", "test", "legacy",
);

check("D4: migrator output re-validates as v1_valid", () => {
  if (!existsSync(SIM_BINARY)) {
    console.log("  skip: bin/nimbus_sim not built");
    return;
  }
  if (!existsSync(LEGACY_DIR)) {
    console.log("  skip: tests/fixtures/configs/test/legacy/ not present");
    return;
  }
  // Pick a couple of v0 fixtures so the test stays fast; A4's bulk
  // round-trip already covers every file in legacy/. This just smokes
  // the loop end-to-end as the Tauri command will exercise it.
  const candidates = ["cartesian_hires.yaml", "cgrid_smoke.yaml"];
  const fixtures = candidates.filter((name: string) =>
    existsSync(join(LEGACY_DIR, name)),
  );
  assertTrue(
    fixtures.length > 0,
    "at least one legacy v0 fixture available",
  );

  for (const fixture of fixtures) {
    const fixturePath = join(LEGACY_DIR, fixture);
    let stdout: string;
    try {
      stdout = execFileSync(
        SIM_BINARY,
        ["--migrate-config", fixturePath],
        { encoding: "utf-8", stdio: ["ignore", "pipe", "ignore"] },
      );
    } catch (e) {
      failed++;
      failures.push(
        `D4 migrate_config(${fixture}): subprocess failed — ${(e as Error).message}`,
      );
      continue;
    }
    assertTrue(
      stdout.length > 0,
      `D4: migrator emitted non-empty stdout for ${fixture}`,
    );
    let migrated: unknown;
    try {
      migrated = parseYaml(stdout);
    } catch (e) {
      failed++;
      failures.push(
        `D4 migrate_config(${fixture}): output is not valid YAML — ${(e as Error).message}`,
      );
      continue;
    }
    const r = validateSchemaVersion(migrated);
    assertEq(
      r.status,
      "v1_valid",
      `D4: migrated ${fixture} re-validates (got ${r.status}: ${summarize(r)})`,
    );
    assertEq(r.schemaStamp, SCHEMA_V1_STAMP, `D4: ${fixture} stamp`);
    assertEq(isBlocking(r), false, `D4: ${fixture} non-blocking`);
  }
});

// ---------- Report ----------

console.log(`\nSchema validator tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) {
    console.log("  " + f);
  }
  process.exit(1);
}
process.exit(0);
