/*
 * Tests for the persistence contract behind the Settings page's dirty state.
 *
 * The page holds edits in component state and only writes them when Save is
 * clicked, and it unmounts on navigation -- so an unsaved edit was silently
 * discarded with nothing on screen distinguishing it from a saved one. The
 * Save button now reports both halves honestly, which rests on two guarantees
 * exercised here:
 *   1. studioSettingsEqual compares every persisted key, so no field can be
 *      edited without the dirty marker noticing;
 *   2. saveStudioSettings REPORTS whether the write landed. It used to swallow
 *      a storage failure and return void, so the button rendered "Saved" over a
 *      write that never happened.
 *
 * Run via:
 *   cd ui && npm run test:studio-settings
 */

import {
  effectiveRunsDirectory,
  STUDIO_SETTINGS_DEFAULTS,
  loadStudioSettings,
  normalizeStudioSettings,
  saveStudioSettings,
  studioSettingsEqual,
  type StudioSettings,
} from "../../../../ui/src/lib/studio/studioSettings.ts";

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

/* ---------- Minimal localStorage + window stand-in ---------- */

interface FakeStorage {
  store: Map<string, string>;
  failWrites: boolean;
}

function installWindow(): FakeStorage {
  const state: FakeStorage = { store: new Map(), failWrites: false };
  const storage = {
    getItem: (key: string) => state.store.get(key) ?? null,
    setItem: (key: string, value: string) => {
      if (state.failWrites) throw new Error("QuotaExceededError");
      state.store.set(key, value);
    },
    removeItem: (key: string) => void state.store.delete(key),
  };
  // saveStudioSettings announces the change to other surfaces; the payload
  // shape does not matter here, only that constructing it cannot throw.
  class StubCustomEvent {
    type: string;
    detail: unknown;
    constructor(type: string, init?: { detail?: unknown }) {
      this.type = type;
      this.detail = init?.detail;
    }
  }
  (globalThis as Record<string, unknown>).window = {
    localStorage: storage,
    dispatchEvent: () => true,
  };
  (globalThis as Record<string, unknown>).CustomEvent =
    (globalThis as Record<string, unknown>).CustomEvent ?? StubCustomEvent;
  return state;
}

const storage = installWindow();

/* ---------- One effective Runs root across Settings, launch, and discovery ---------- */

check(
  "an empty Runs directory resolves to the Nimbus default",
  effectiveRunsDirectory("") === STUDIO_SETTINGS_DEFAULTS.runsDirectory,
);
check(
  "a whitespace Runs directory resolves to the Nimbus default",
  effectiveRunsDirectory("  \n ") === STUDIO_SETTINGS_DEFAULTS.runsDirectory,
);
check(
  "a configured Runs directory is trimmed once",
  effectiveRunsDirectory("  ~/research/runs  ") === "~/research/runs",
);
check(
  "normalization cannot persist a process-working-directory archive root",
  normalizeStudioSettings({ runsDirectory: " " }).runsDirectory
    === STUDIO_SETTINGS_DEFAULTS.runsDirectory,
);

/* ---------- studioSettingsEqual covers every persisted key ---------- */

const base: StudioSettings = { ...STUDIO_SETTINGS_DEFAULTS };
check("a settings record equals itself", studioSettingsEqual(base, { ...base }));

const keys = Object.keys(STUDIO_SETTINGS_DEFAULTS) as Array<keyof StudioSettings>;
check("the defaults expose more than one key", keys.length > 1);

// Every key must be load-bearing. A comparison that skipped one would let an
// edit to that field look saved.
for (const key of keys) {
  const mutated: StudioSettings = { ...base };
  mutated[key] = (key === "appearance" ? "light" : "__changed__") as never;
  check(`editing ${key} registers as dirty`, !studioSettingsEqual(base, mutated));
}

/* ---------- saveStudioSettings reports whether the write landed ---------- */

const edited: StudioSettings = { ...base, runsDirectory: "~/elsewhere/runs" };
check("a successful save reports true", saveStudioSettings(edited) === true);
check("the value round-trips through storage", loadStudioSettings().runsDirectory === "~/elsewhere/runs");
check("a saved record is no longer dirty", studioSettingsEqual(loadStudioSettings(), edited));

storage.failWrites = true;
const unwritable: StudioSettings = { ...edited, binaryPathOverride: "/opt/nimbus/bin/nimbus_sim" };
check("a failed save reports false", saveStudioSettings(unwritable) === false);
check(
  "a failed save leaves the stored record untouched",
  loadStudioSettings().binaryPathOverride === "",
);
// The caller keeps showing "Unsaved changes" off this comparison.
check("a failed save stays dirty", !studioSettingsEqual(loadStudioSettings(), unwritable));
storage.failWrites = false;

/* ---------- Reading back a corrupt record cannot fabricate a clean state ---------- */

storage.store.set("nimbus-studio:settings:v1", "{not json");
check("unparseable storage falls back to defaults", studioSettingsEqual(loadStudioSettings(), STUDIO_SETTINGS_DEFAULTS));
check(
  "a partial record normalizes to a full one",
  studioSettingsEqual(
    normalizeStudioSettings({ runsDirectory: "~/partial" }),
    { ...STUDIO_SETTINGS_DEFAULTS, runsDirectory: "~/partial" },
  ),
);

/* ---------- Plot side ---------- */

check(
  "plot side defaults to right",
  STUDIO_SETTINGS_DEFAULTS.plotSide === "right",
);
check(
  "a stored plot side round-trips",
  normalizeStudioSettings({ plotSide: "left" }).plotSide === "left",
);
/* An unrecognised value must fall back rather than reach the DOM: it is written
 * straight into a data attribute that CSS selects on, so a junk value would
 * silently produce neither layout. */
check(
  "an unknown plot side falls back to the default",
  normalizeStudioSettings({ plotSide: "sideways" }).plotSide === "right",
);
check(
  "a record predating the setting migrates to the default",
  normalizeStudioSettings({ runsDirectory: "~/legacy" }).plotSide === "right",
);
check(
  "plot side participates in dirty-state comparison",
  !studioSettingsEqual(
    STUDIO_SETTINGS_DEFAULTS,
    { ...STUDIO_SETTINGS_DEFAULTS, plotSide: "left" },
  ),
);

console.log(`\nstudio-settings tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  " + f);
  process.exit(1);
}
process.exit(0);
