/*
 * Tests for `src/lib/plot/cameraOrientation.ts` — the pure camera-orientation
 * math used by the 3D compass widget.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types --import ../tests/ui/register-ts-resolve.mjs \
 *             ../tests/ui/unit/plot/test_camera_orientation.ts
 */

import {
  computeCameraOrientation,
  compassOctant,
  type Vec3M,
} from "../../../../ui/src/lib/plot/cameraOrientation.ts";

// ---------- Tiny assertion API (mirrors test_quiver.ts) ----------

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

function approx(a: number, b: number, eps = 0.5): boolean {
  return Math.abs(a - b) <= eps;
}

function check(label: string, fn: () => void): void {
  try {
    fn();
  } catch (e) {
    failed++;
    failures.push(`THREW ${label}: ${(e as Error).message}`);
  }
}

// ---------- Shared test geometry ----------

// All cameras are at the origin looking toward four cardinal directions.
// The scene convention: x = east, y = north, z = up.
const ORIGIN: Vec3M = [0, 0, 0];
const UP: Vec3M = [0, 0, 1];

const NORTH_FOCAL: Vec3M = [0, 1000, 0]; // looking north (+y)
const EAST_FOCAL: Vec3M = [1000, 0, 0]; // looking east  (+x)
const SOUTH_FOCAL: Vec3M = [0, -1000, 0]; // looking south (-y)
const WEST_FOCAL: Vec3M = [-1000, 0, 0]; // looking west  (-x)

// ---------- 1. Cardinal azimuth checks (±0.5° tolerance) ----------
// Four checks are required so a single sign flip or argument-order swap
// cannot accidentally pass all tests.

check("looking north → azimuth ≈ 0°", () => {
  const result = computeCameraOrientation(ORIGIN, NORTH_FOCAL, UP);
  assertTrue(result !== null, "north: result is not null");
  assertTrue(approx(result!.azimuthDeg, 0), `north: azimuth=${result!.azimuthDeg}`);
});

check("looking east → azimuth ≈ 90°", () => {
  const result = computeCameraOrientation(ORIGIN, EAST_FOCAL, UP);
  assertTrue(result !== null, "east: result is not null");
  assertTrue(approx(result!.azimuthDeg, 90), `east: azimuth=${result!.azimuthDeg}`);
});

check("looking south → azimuth ≈ 180°", () => {
  const result = computeCameraOrientation(ORIGIN, SOUTH_FOCAL, UP);
  assertTrue(result !== null, "south: result is not null");
  assertTrue(approx(result!.azimuthDeg, 180), `south: azimuth=${result!.azimuthDeg}`);
});

check("looking west → azimuth ≈ 270°", () => {
  const result = computeCameraOrientation(ORIGIN, WEST_FOCAL, UP);
  assertTrue(result !== null, "west: result is not null");
  assertTrue(approx(result!.azimuthDeg, 270), `west: azimuth=${result!.azimuthDeg}`);
});

// ---------- 2. Elevation checks ----------

check("looking straight down → elevation ≈ −90°", () => {
  const result = computeCameraOrientation([0, 0, 1000], [0, 0, 0], UP);
  assertTrue(result !== null, "down: result is not null");
  assertTrue(approx(result!.elevationDeg, -90), `down: elevation=${result!.elevationDeg}`);
});

check("looking straight up → elevation ≈ +90°", () => {
  const result = computeCameraOrientation([0, 0, 0], [0, 0, 1000], UP);
  assertTrue(result !== null, "up: result is not null");
  assertTrue(approx(result!.elevationDeg, 90), `up: elevation=${result!.elevationDeg}`);
});

check("looking due north at 0° elevation → elevation ≈ 0°", () => {
  const result = computeCameraOrientation(ORIGIN, NORTH_FOCAL, UP);
  assertTrue(result !== null, "level: result is not null");
  assertTrue(approx(result!.elevationDeg, 0), `level: elevation=${result!.elevationDeg}`);
});

check("looking 45° up toward north → elevation ≈ +45°", () => {
  // Equal horizontal and vertical displacement gives 45°.
  const result = computeCameraOrientation([0, 0, 0], [0, 1000, 1000], UP);
  assertTrue(result !== null, "45up: result is not null");
  assertTrue(approx(result!.elevationDeg, 45), `45up: elevation=${result!.elevationDeg}`);
});

// ---------- 3. Storm bearing and range ----------

check("storm directly north at 5 km range", () => {
  const storm: Vec3M = [0, 5000, 0];
  const result = computeCameraOrientation(ORIGIN, EAST_FOCAL, UP, storm);
  assertTrue(result !== null, "storm-north: result is not null");
  assertTrue(
    approx(result!.stormBearingDeg!, 0),
    `storm-north: bearing=${result!.stormBearingDeg}`,
  );
  assertTrue(
    approx(result!.stormRangeM!, 5000, 1),
    `storm-north: range=${result!.stormRangeM}`,
  );
});

check("storm to the south-east at known offset", () => {
  // Storm at x=+3000, y=−3000 → bearing = 135° (SE).
  const storm: Vec3M = [3000, -3000, 0];
  const result = computeCameraOrientation(ORIGIN, NORTH_FOCAL, UP, storm);
  assertTrue(result !== null, "storm-SE: result is not null");
  assertTrue(
    approx(result!.stormBearingDeg!, 135),
    `storm-SE: bearing=${result!.stormBearingDeg}`,
  );
  // Horizontal range = √(3000²+3000²) ≈ 4243 m.
  const expectedRange = Math.hypot(3000, 3000);
  assertTrue(
    approx(result!.stormRangeM!, expectedRange, 1),
    `storm-SE: range=${result!.stormRangeM}, expected≈${expectedRange.toFixed(0)}`,
  );
});

check("storm range is horizontal even when the camera is elevated", () => {
  const camera: Vec3M = [0, 0, 8000];
  const storm: Vec3M = [3000, 4000, 0];
  const result = computeCameraOrientation(camera, [0, 1000, 8000], UP, storm);
  assertTrue(result !== null, "elevated-camera: result is not null");
  assertTrue(
    approx(result!.stormRangeM!, 5000, 1),
    `elevated-camera: horizontal range=${result!.stormRangeM}`,
  );
});

// ---------- 4. compassOctant boundaries ----------

check("compassOctant: exact cardinal directions", () => {
  assertTrue(compassOctant(0) === "N", "0° = N");
  assertTrue(compassOctant(90) === "E", "90° = E");
  assertTrue(compassOctant(180) === "S", "180° = S");
  assertTrue(compassOctant(270) === "W", "270° = W");
});

check("compassOctant: intercardinal and intermediate sectors", () => {
  assertTrue(compassOctant(45) === "NE", "45° = NE");
  assertTrue(compassOctant(135) === "SE", "135° = SE");
  assertTrue(compassOctant(225) === "SW", "225° = SW");
  assertTrue(compassOctant(315) === "NW", "315° = NW");
  assertTrue(compassOctant(22) === "NNE", "22° = NNE");
  assertTrue(compassOctant(337) === "NNW", "337° = NNW");
});

check("compassOctant: wrap across 0°/360° boundary", () => {
  // 354° is between 348.75° (NNW midpoint) and 360°/0° (N midpoint).
  // Boundary between NNW and N is at 348.75°, so 354° → N (closer to 360/0).
  assertTrue(compassOctant(354) === "N", `354° = N, got ${compassOctant(354)}`);
  // 360° wraps to 0° = N.
  assertTrue(compassOctant(360) === "N", `360° wraps to N, got ${compassOctant(360)}`);
  // 359.9° is still N.
  assertTrue(compassOctant(359.9) === "N", `359.9° = N, got ${compassOctant(359.9)}`);
  // 348.74° is just inside NNW.
  assertTrue(compassOctant(348.74) === "NNW", `348.74° = NNW, got ${compassOctant(348.74)}`);
});

check("compassOctant: non-finite input returns N without throwing", () => {
  assertTrue(compassOctant(NaN) === "N", "NaN → N");
  assertTrue(compassOctant(Infinity) === "N", "Infinity → N");
  assertTrue(compassOctant(-Infinity) === "N", "-Infinity → N");
});

// ---------- 5. Degenerate inputs → null ----------

check("coincident position and focal point → null", () => {
  const result = computeCameraOrientation([100, 200, 300], [100, 200, 300], UP);
  assertTrue(result === null, "coincident: returns null");
});

check("non-finite position → null", () => {
  const result = computeCameraOrientation([NaN, 0, 0], NORTH_FOCAL, UP);
  assertTrue(result === null, "NaN position: returns null");
});

check("non-finite focal point → null", () => {
  const result = computeCameraOrientation(ORIGIN, [0, Infinity, 0], UP);
  assertTrue(result === null, "Inf focal: returns null");
});

check("non-finite view-up → null", () => {
  const result = computeCameraOrientation(ORIGIN, NORTH_FOCAL, [0, NaN, 0]);
  assertTrue(result === null, "NaN viewUp: returns null");
});

check("non-finite storm centre → null", () => {
  const storm: Vec3M = [NaN, 0, 0];
  const result = computeCameraOrientation(ORIGIN, NORTH_FOCAL, UP, storm);
  assertTrue(result === null, "NaN stormCentre: returns null");
});

// ---------- 6. Vertical-exaggeration neutrality ----------
//
// The API takes TRUE METRES. A caller must undo the scene scale before
// passing positions. Identical true-metre geometry must yield identical
// results regardless of what scale factor the caller applied and then undid.
//
// This test demonstrates the correct call pattern and pins the invariant:
// if a caller forgot to undo scale.z=3 the elevation would read as ~72° instead
// of ~45°, so checking the exact value also verifies the formula is correct.

check("undoing vertical exaggeration preserves elevation", () => {
  // True-metre geometry: camera at (0, 0, 1000), looking at (0, 1000, 1000).
  // Horizontal: 1000 m north. Vertical: 0 m. → elevation = 0°.
  const posTrue: Vec3M = [0, 0, 1000];
  const focalTrue: Vec3M = [0, 1000, 1000];

  // Result with true metres (no scaling involved).
  const resultTrue = computeCameraOrientation(posTrue, focalTrue, UP);
  assertTrue(resultTrue !== null, "true-metre: result is not null");
  assertTrue(
    approx(resultTrue!.elevationDeg, 0),
    `true-metre elevation=${resultTrue!.elevationDeg} expected 0`,
  );

  // Simulate a 3× vertical exaggeration: what the camera stores.
  // Camera position in scene: z *= 3.
  const scaleZ = 3;
  const posScene: Vec3M = [posTrue[0], posTrue[1], posTrue[2] * scaleZ];
  const focalScene: Vec3M = [focalTrue[0], focalTrue[1], focalTrue[2] * scaleZ];

  // Caller undoes scale before calling (correct usage).
  const posUndone: Vec3M = [posScene[0], posScene[1], posScene[2] / scaleZ];
  const focalUndone: Vec3M = [focalScene[0], focalScene[1], focalScene[2] / scaleZ];
  const resultUndone = computeCameraOrientation(posUndone, focalUndone, UP);
  assertTrue(resultUndone !== null, "undone-scale: result is not null");
  assertTrue(
    approx(resultUndone!.elevationDeg, 0),
    `undone-scale elevation=${resultUndone!.elevationDeg} expected 0`,
  );

  // Caller does NOT undo scale (incorrect usage — elevation would be wrong).
  const resultExag = computeCameraOrientation(posScene, focalScene, UP);
  assertTrue(resultExag !== null, "exag: result is not null");
  // With z=3x the camera appears 3000 m up but focal is also at 3000 m z,
  // so the view vector z-component is 0 in this particular example — elevation
  // stays 0 only because the z displacement cancels. Let's use a camera where
  // the look-down IS affected by exaggeration.
  //
  // Real test: camera at (0, 0, 0), focal at (0, 1000, 1000).
  // True elevation = atan2(1000, 1000) = 45°.
  // With 3× exag, scene focal = (0, 1000, 3000).
  // Scene elevation = atan2(3000, 1000) ≈ 71.6°. Must NOT equal true.
  const scaleZAsym = 3;
  const posA: Vec3M = [0, 0, 0];
  const focalTrueA: Vec3M = [0, 1000, 1000];
  const focalSceneA: Vec3M = [0, 1000, focalTrueA[2] * scaleZAsym];

  const resultTrueA = computeCameraOrientation(posA, focalTrueA, UP);
  assertTrue(resultTrueA !== null, "asym-true: result is not null");
  assertTrue(
    approx(resultTrueA!.elevationDeg, 45),
    `asym-true: elevation=${resultTrueA!.elevationDeg} expected 45`,
  );

  const resultExagA = computeCameraOrientation(posA, focalSceneA, UP);
  assertTrue(resultExagA !== null, "asym-exag: result is not null");
  const exagElevation = resultExagA!.elevationDeg;
  assertTrue(
    !approx(exagElevation, 45, 2),
    `asym-exag: elevation ${exagElevation.toFixed(1)}° should differ from 45° (undoing exag matters)`,
  );

  // Caller undoes scale correctly → gets the true 45°.
  const focalUndoneA: Vec3M = [focalSceneA[0], focalSceneA[1], focalSceneA[2] / scaleZAsym];
  const resultUndoneA = computeCameraOrientation(posA, focalUndoneA, UP);
  assertTrue(resultUndoneA !== null, "asym-undone: result is not null");
  assertTrue(
    approx(resultUndoneA!.elevationDeg, 45),
    `asym-undone: elevation=${resultUndoneA!.elevationDeg} expected 45`,
  );
});

// ---------- 7. Azimuth is in [0, 360) for all inputs ----------

check("azimuth is always non-negative and below 360°", () => {
  // Camera behind the origin looking in each diagonal to stress normalisation.
  const diagonals: Array<[Vec3M, number]> = [
    [[1000, 1000, 0], 45], // NE
    [[-1000, 1000, 0], 315], // NW
    [[-1000, -1000, 0], 225], // SW
    [[1000, -1000, 0], 135], // SE
  ];
  for (const [focal, expected] of diagonals) {
    const r = computeCameraOrientation(ORIGIN, focal, UP);
    assertTrue(r !== null, `diagonal ${expected}°: not null`);
    assertTrue(
      r!.azimuthDeg >= 0 && r!.azimuthDeg < 360,
      `diagonal ${expected}°: azimuth=${r!.azimuthDeg} in [0,360)`,
    );
    assertTrue(
      approx(r!.azimuthDeg, expected),
      `diagonal ${expected}°: azimuth=${r!.azimuthDeg}`,
    );
  }
});

// ---------- Report ----------

console.log(`\ncameraOrientation tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  " + f);
  process.exit(1);
}
process.exit(0);
