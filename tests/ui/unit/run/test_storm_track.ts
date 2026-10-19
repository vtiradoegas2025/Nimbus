/*
 * Tests for src/lib/run/stormTrack.ts.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types \
 *     --import ../tests/ui/register-ts-resolve.mjs \
 *     ../tests/ui/unit/run/test_storm_track.ts
 *
 * No test framework — uses the same tiny inline assertion API as test_quiver.ts.
 */

import {
  parseStormTrack,
  stormTrackCentre,
  stormTrackMatchesFrame,
} from "../../../../ui/src/lib/run/stormTrack.ts";

// ---------- Tiny assertion API (matches test_quiver.ts) ----------

let failed = 0;
let passed = 0;
const failures: string[] = [];

function assertTrue(cond: boolean, label: string): void {
  if (cond) passed++;
  else {
    failed++;
    failures.push(`FAIL  ${label}`);
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

// ---------- Minimal valid storm-track JSON (schema v9) ----------

function makeValidJson(overrides: Record<string, unknown> = {}): string {
  const base: Record<string, unknown> = {
    schema: "nimbus.storm_feature",
    schema_version: 9,
    completed_step: 142,
    simulation_time_s: 710.0,
    selected_feature_available: true,
    maximum_absolute_updraft_helicity_m2s2: 3142.7,
    grid_centroid_m: { x: 12345.0, y: 6789.0 },
    grid_core_centroid_m: { x: 12200.0, y: 6900.0 },
    centroid_m: { x: 112345.0, y: 106789.0 },        // earth-relative (offset by 100000 m)
    core_centroid_m: { x: 112200.0, y: 106900.0 },   // earth-relative
    branch_candidates: [
      {
        rank: 0,
        rotation_class: "strongly_rotating",
        maximum_updraft_ms: 45.2,
      },
    ],
  };
  return JSON.stringify({ ...base, ...overrides });
}

// ---------- 1. Valid JSON parses successfully ----------

check("Valid JSON: parse succeeds and snapshot fields match", () => {
  const result = parseStormTrack(makeValidJson());
  assertTrue(result.ok, "parse succeeds on valid JSON");
  if (!result.ok) return;

  const s = result.snapshot;
  assertTrue(s.schema === "nimbus.storm_feature", "schema field preserved");
  assertTrue(s.schemaVersion === 9, "schemaVersion = 9");
  assertTrue(s.completedStep === 142, "completedStep = 142");
  assertTrue(s.simulationTimeS === 710.0, "simulationTimeS = 710.0");
  assertTrue(s.featureAvailable === true, "featureAvailable = true");
  assertTrue(
    s.earthCoreCentroidM?.xM === 112200 && s.earthCoreCentroidM?.yM === 106900,
    "Earth core centroid is retained for world-coordinate rendering",
  );
  assertTrue(
    Math.abs((s.maximumAbsoluteUpdraftHelicityM2S2 ?? 0) - 3142.7) < 0.01,
    `max |UH| ≈ 3142.7 (got ${s.maximumAbsoluteUpdraftHelicityM2S2})`,
  );
  assertTrue(s.rotationClass === "strongly_rotating", `rotation class (got ${s.rotationClass})`);
});

check("3D storm centre uses Earth coordinates, not grid-relative coordinates", () => {
  const result = parseStormTrack(makeValidJson());
  assertTrue(result.ok, "parse succeeds");
  if (!result.ok) return;
  assertTrue(
    stormTrackCentre(result.snapshot)?.join(",") === "112200,106900,0",
    `storm centre matches VTK Earth-world frame (got ${stormTrackCentre(result.snapshot)})`,
  );
});

check("run-level latest snapshot is accepted only for its matching rendered time", () => {
  const result = parseStormTrack(makeValidJson());
  assertTrue(result.ok, "parse succeeds");
  if (!result.ok) return;
  assertTrue(stormTrackMatchesFrame(result.snapshot, 710), "matching time is accepted");
  assertTrue(stormTrackMatchesFrame(result.snapshot, 710 + 1e-7), "float reconstruction noise is accepted");
  assertTrue(!stormTrackMatchesFrame(result.snapshot, 705), "older rendered frame rejects latest marker");
  assertTrue(!stormTrackMatchesFrame(result.snapshot, null), "unknown rendered time fails closed");
});

// ---------- 2. Grid-relative coordinates are selected over earth-origin ones ----------

check("Grid-relative coordinates are used, not earth-origin ones", () => {
  const result = parseStormTrack(makeValidJson());
  assertTrue(result.ok, "parse succeeds");
  if (!result.ok) return;

  const s = result.snapshot;
  // grid_core_centroid_m is at (12200, 6900) — NOT (112200, 106900).
  assertTrue(
    s.gridCoreCentroidM !== null &&
    Math.abs(s.gridCoreCentroidM.xM - 12200) < 1 &&
    Math.abs(s.gridCoreCentroidM.yM - 6900) < 1,
    `grid_core_centroid_m = (12200, 6900) not the earth-relative offset (got ${JSON.stringify(s.gridCoreCentroidM)})`,
  );

  // grid_centroid_m is at (12345, 6789) — NOT (112345, 106789).
  assertTrue(
    s.gridCentroidM !== null &&
    Math.abs(s.gridCentroidM.xM - 12345) < 1 &&
    Math.abs(s.gridCentroidM.yM - 6789) < 1,
    `grid_centroid_m = (12345, 6789) not the earth-relative offset (got ${JSON.stringify(s.gridCentroidM)})`,
  );
});

// ---------- 3. Absent input (tracking disabled) ----------

check("Null input returns ok:false with reason 'absent'", () => {
  const result = parseStormTrack(null);
  assertTrue(!result.ok, "null -> not ok");
  if (result.ok) return;
  assertTrue(result.reason === "absent", `reason = 'absent' (got '${result.reason}')`);
});

check("Undefined input returns ok:false with reason 'absent'", () => {
  const result = parseStormTrack(undefined);
  assertTrue(!result.ok, "undefined -> not ok");
  if (result.ok) return;
  assertTrue(result.reason === "absent", `reason = 'absent' (got '${result.reason}')`);
});

// ---------- 4. Truncated / malformed JSON ----------

check("Truncated JSON string returns ok:false with reason 'json_error'", () => {
  const truncated = makeValidJson().slice(0, 40); // cut off mid-object
  const result = parseStormTrack(truncated);
  assertTrue(!result.ok, "truncated JSON -> not ok");
  if (result.ok) return;
  assertTrue(result.reason === "json_error", `reason = 'json_error' (got '${result.reason}')`);
});

check("Non-object JSON (array) returns ok:false with reason 'json_error'", () => {
  const result = parseStormTrack("[1, 2, 3]");
  assertTrue(!result.ok, "array JSON -> not ok");
  if (result.ok) return;
  assertTrue(result.reason === "json_error", `reason = 'json_error' (got '${result.reason}')`);
});

check("Completely empty string returns ok:false", () => {
  const result = parseStormTrack("");
  assertTrue(!result.ok, "empty string -> not ok");
});

// ---------- 5. Wrong schema identifier ----------

check("Wrong schema returns ok:false with reason 'schema_mismatch'", () => {
  const result = parseStormTrack(makeValidJson({ schema: "something.else" }));
  assertTrue(!result.ok, "wrong schema -> not ok");
  if (result.ok) return;
  assertTrue(result.reason === "schema_mismatch", `reason = 'schema_mismatch' (got '${result.reason}')`);
});

// ---------- 6. Missing required keys ----------

check("Missing schema_version returns ok:false with reason 'missing_keys'", () => {
  const obj = JSON.parse(makeValidJson());
  delete (obj as Record<string, unknown>)["schema_version"];
  const result = parseStormTrack(obj);
  assertTrue(!result.ok, "missing schema_version -> not ok");
  if (result.ok) return;
  assertTrue(result.reason === "missing_keys", `reason = 'missing_keys' (got '${result.reason}')`);
});

check("Missing completed_step returns ok:false with reason 'missing_keys'", () => {
  const obj = JSON.parse(makeValidJson());
  delete (obj as Record<string, unknown>)["completed_step"];
  const result = parseStormTrack(obj);
  assertTrue(!result.ok, "missing completed_step -> not ok");
  if (result.ok) return;
  assertTrue(result.reason === "missing_keys", `reason = 'missing_keys' (got '${result.reason}')`);
});

check("Missing simulation_time_s returns ok:false with reason 'missing_keys'", () => {
  const obj = JSON.parse(makeValidJson());
  delete (obj as Record<string, unknown>)["simulation_time_s"];
  const result = parseStormTrack(obj);
  assertTrue(!result.ok, "missing simulation_time_s -> not ok");
  if (result.ok) return;
  assertTrue(result.reason === "missing_keys", `reason = 'missing_keys' (got '${result.reason}')`);
});

// ---------- 7. No feature available: optional fields are null ----------

check("No feature (selected_feature_available=false): optional fields are null", () => {
  const result = parseStormTrack(makeValidJson({ selected_feature_available: false }));
  assertTrue(result.ok, "parse succeeds when feature absent");
  if (!result.ok) return;

  const s = result.snapshot;
  assertTrue(s.featureAvailable === false, "featureAvailable = false");
  assertTrue(s.earthCoreCentroidM === null, "earthCoreCentroidM is null");
  assertTrue(s.earthCentroidM === null, "earthCentroidM is null");
  assertTrue(s.gridCoreCentroidM === null, "gridCoreCentroidM is null");
  assertTrue(s.gridCentroidM === null, "gridCentroidM is null");
  assertTrue(s.maximumAbsoluteUpdraftHelicityM2S2 === null, "max |UH| is null");
  assertTrue(s.rotationClass === null, "rotationClass is null");
});

// ---------- 8. Accepts a pre-parsed object (not just a string) ----------

check("Accepts a pre-parsed object", () => {
  const obj = JSON.parse(makeValidJson());
  const result = parseStormTrack(obj);
  assertTrue(result.ok, "pre-parsed object -> ok");
});

// ---------- 9. Partial grid_core_centroid_m (missing y): field is null ----------

check("Malformed grid_core_centroid_m (missing y) -> gridCoreCentroidM null", () => {
  const result = parseStormTrack(
    makeValidJson({ grid_core_centroid_m: { x: 5000 } }) // y missing
  );
  assertTrue(result.ok, "parse still succeeds");
  if (!result.ok) return;
  assertTrue(
    result.snapshot.gridCoreCentroidM === null,
    "gridCoreCentroidM is null when y key is absent",
  );
});

// ---------- 10. No branch_candidates: rotationClass is null ----------

check("No branch_candidates array: rotationClass is null", () => {
  const obj = JSON.parse(makeValidJson());
  delete (obj as Record<string, unknown>)["branch_candidates"];
  const result = parseStormTrack(obj);
  assertTrue(result.ok, "parse succeeds without branch_candidates");
  if (!result.ok) return;
  assertTrue(result.snapshot.rotationClass === null, "rotationClass null when candidates absent");
});

// ---------- 11. Empty branch_candidates: rotationClass is null ----------

check("Empty branch_candidates: rotationClass is null", () => {
  const result = parseStormTrack(makeValidJson({ branch_candidates: [] }));
  assertTrue(result.ok, "parse succeeds with empty candidates array");
  if (!result.ok) return;
  assertTrue(result.snapshot.rotationClass === null, "rotationClass null for empty candidates");
});

// ---------- Report ----------

console.log(`\nstorm-track tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  " + f);
  process.exit(1);
}
process.exit(0);
