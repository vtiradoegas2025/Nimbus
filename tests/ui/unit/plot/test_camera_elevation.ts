/*
 * Vertical camera placement for `src/lib/plot/cameraElevation.ts`.
 *
 * The property that matters is that a height means the same thing on screen as
 * it does in the sounding: asking for 500 m must put the focal point at 500 m
 * above the domain floor, under any vertical exaggeration. The rest guards the
 * degenerate cases a camera cannot survive — a view direction parallel to the
 * up vector, and a position coincident with its own focal point.
 *
 * Run via:
 *   cd ui && npm run test:camera-elevation
 */

import {
  CAMERA_BEARING_PRESETS,
  CAMERA_HEIGHT_STEP_M,
  CAMERA_VIEW_ANGLE_DEGREES,
  DEFAULT_AZIMUTH_RADIANS,
  DEFAULT_ELEVATION_DEGREES,
  ELEVATION_PRESETS,
  cameraAzimuthRadians,
  cameraPositionAzimuthForBearing,
  cameraPlacementAtHeight,
  fitDistanceM,
  clampElevationDegrees,
  clampHeightM,
  defaultCameraHeightM,
  domainHeightM,
} from "../../../../ui/src/lib/plot/cameraElevation.ts";
import type { DomainBoundsM } from "../../../../ui/src/lib/plot/domainAxes.ts";

// ---------- Tiny assertion API ----------

let failed = 0;
let passed = 0;
const failures: string[] = [];

function assertTrue(cond: boolean, label: string): void {
  if (cond) {
    passed++;
  } else {
    failed++;
    failures.push(`FAIL ${label}`);
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

const DOMAIN: DomainBoundsM = {
  lower: [-40000, -30000, 0],
  upper: [40000, 30000, 18000],
};

/** A domain whose floor is not at sea level, so "above ground" is not "above zero". */
const ELEVATED: DomainBoundsM = {
  lower: [-40000, -30000, 1200],
  upper: [40000, 30000, 19200],
};

const UNIT = { xy: 1, z: 1 };

function magnitude(a: readonly number[], b: readonly number[]): number {
  return Math.hypot(a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!);
}

// ---------- Domain height ----------

check("domain height is the vertical span", () => {
  assertTrue(domainHeightM(DOMAIN) === 18000, "flat-floored domain height");
  assertTrue(domainHeightM(ELEVATED) === 18000, "raised-floor domain height");
  assertTrue(domainHeightM(null) === null, "no bounds yields no height");
});

check("a collapsed domain reports no height rather than zero", () => {
  const flat: DomainBoundsM = { lower: [0, 0, 500], upper: [1, 1, 500] };
  assertTrue(domainHeightM(flat) === null, "zero-thickness domain has no usable height");
});

check("the opening height sits in the mid-levels", () => {
  const height = defaultCameraHeightM(DOMAIN);
  assertTrue(height === 6000, `opening height is a third of the domain (${height})`);
  assertTrue(defaultCameraHeightM(null) === 0, "no bounds opens at the floor");
});

// ---------- Clamping ----------

check("heights are clamped into their domain", () => {
  assertTrue(clampHeightM(-500, DOMAIN) === 0, "below ground clamps to the floor");
  assertTrue(clampHeightM(99000, DOMAIN) === 18000, "above the lid clamps to the lid");
  assertTrue(clampHeightM(2500, DOMAIN) === 2500, "an in-range height is untouched");
  assertTrue(clampHeightM(Number.NaN, DOMAIN) === 0, "NaN falls back to the floor");
});

check("look angles are clamped away from the downward pole", () => {
  assertTrue(clampElevationDegrees(90) === 90, "straight down is reachable");
  assertTrue(clampElevationDegrees(140) === 90, "past vertical clamps to vertical");
  assertTrue(clampElevationDegrees(-120) === -89, "from below stops short of the pole");
  assertTrue(
    clampElevationDegrees(Number.NaN) === DEFAULT_ELEVATION_DEGREES,
    "NaN falls back to the opening angle",
  );
});

// ---------- Height is honoured ----------

check("the focal point sits at the requested height above the domain floor", () => {
  for (const heightM of [0, 500, 3000, 12000]) {
    const placement = cameraPlacementAtHeight(DOMAIN, UNIT, heightM, 0, 0);
    assertTrue(placement != null, `height ${heightM} yields a placement`);
    if (!placement) continue;
    assertTrue(
      Math.abs(placement.focalPoint[2] - heightM) < 1e-9,
      `focal point at ${heightM} m (got ${placement.focalPoint[2]})`,
    );
  }
});

check("height is measured above the domain floor, not above zero", () => {
  const placement = cameraPlacementAtHeight(ELEVATED, UNIT, 500, 0, 0);
  if (!placement) return;
  assertTrue(
    Math.abs(placement.focalPoint[2] - 1700) < 1e-9,
    `500 m above a 1200 m floor is 1700 m (got ${placement.focalPoint[2]})`,
  );
});

check("vertical exaggeration moves the focal point with the data it is framing", () => {
  for (const z of [1, 2.5, 6]) {
    const placement = cameraPlacementAtHeight(DOMAIN, { xy: 1, z }, 4000, 0, 0);
    if (!placement) continue;
    assertTrue(
      Math.abs(placement.focalPoint[2] - 4000 * z) < 1e-9,
      `at ${z}x exaggeration 4 km draws at ${4000 * z} (got ${placement.focalPoint[2]})`,
    );
  }
});

check("the focal point is centred horizontally under any horizontal scale", () => {
  const offset: DomainBoundsM = { lower: [0, 0, 0], upper: [80000, 60000, 18000] };
  for (const xy of [0.5, 1, 2]) {
    const placement = cameraPlacementAtHeight(offset, { xy, z: 1 }, 1000, 30, 0);
    if (!placement) continue;
    assertTrue(
      Math.abs(placement.focalPoint[0] - 40000 * xy) < 1e-9
      && Math.abs(placement.focalPoint[1] - 30000 * xy) < 1e-9,
      `horizontal scale ${xy} centres the focal point`,
    );
  }
});

check("an in-domain meteorological target replaces the horizontal domain centre", () => {
  const target = [12000, -9000] as const;
  const placement = cameraPlacementAtHeight(DOMAIN, { xy: 2, z: 1 }, 1000, 30, 0, {
    targetXYM: target,
  });
  if (!placement) return;
  assertTrue(
    placement.focalPoint[0] === target[0] * 2 && placement.focalPoint[1] === target[1] * 2,
    "the camera focuses the tracked storm instead of empty domain centre",
  );
});

check("an out-of-domain meteorological target is clamped to the physical domain", () => {
  const placement = cameraPlacementAtHeight(DOMAIN, UNIT, 1000, 30, 0, {
    targetXYM: [90000, -90000],
  });
  if (!placement) return;
  assertTrue(
    placement.focalPoint[0] === DOMAIN.upper[0] && placement.focalPoint[1] === DOMAIN.lower[1],
    "the focal target cannot escape the published domain",
  );
});

// ---------- Look angle ----------

check("a level look angle puts the camera at the height it is looking at", () => {
  const placement = cameraPlacementAtHeight(DOMAIN, UNIT, 700, 0, 0);
  if (!placement) return;
  assertTrue(
    Math.abs(placement.position[2] - placement.focalPoint[2]) < 1e-9,
    "level means eye height equals target height",
  );
});

check("a steeper look angle raises the camera without moving the target", () => {
  const level = cameraPlacementAtHeight(DOMAIN, UNIT, 700, 0, 0);
  const oblique = cameraPlacementAtHeight(DOMAIN, UNIT, 700, 30, 0);
  if (!level || !oblique) return;
  assertTrue(oblique.position[2] > level.position[2], "30 degrees sits above level");
  assertTrue(
    Math.abs(oblique.focalPoint[2] - level.focalPoint[2]) < 1e-9,
    "the target height is unchanged by the look angle",
  );
});

check("a negative look angle places the camera below its target", () => {
  const placement = cameraPlacementAtHeight(DOMAIN, UNIT, 8000, -20, 0);
  if (!placement) return;
  assertTrue(placement.position[2] < placement.focalPoint[2], "looking up from below");
});

check("every preset yields a usable placement", () => {
  for (const preset of ELEVATION_PRESETS) {
    const placement = cameraPlacementAtHeight(DOMAIN, UNIT, 2000, preset.degrees, 0);
    assertTrue(placement != null, `${preset.id} yields a placement`);
    if (!placement) continue;
    const distance = magnitude(placement.position, placement.focalPoint);
    assertTrue(distance > 0, `${preset.id} keeps the camera off its own focal point`);
    const up = placement.viewUp;
    const view = [0, 1, 2].map((i) => placement.focalPoint[i]! - placement.position[i]!);
    const viewLength = Math.hypot(view[0]!, view[1]!, view[2]!);
    const upLength = Math.hypot(up[0], up[1], up[2]);
    const alignment = Math.abs(
      (view[0]! * up[0] + view[1]! * up[1] + view[2]! * up[2]) / (viewLength * upLength),
    );
    assertTrue(alignment < 0.999, `${preset.id} keeps the up vector off the view axis`);
  }
});

check("looking straight down rolls the up vector off the vertical", () => {
  const top = cameraPlacementAtHeight(DOMAIN, UNIT, 2000, 90, 0);
  if (!top) return;
  assertTrue(top.viewUp[2] === 0, "a plan view cannot keep +z up");
  assertTrue(top.viewUp[1] === 1, "north stays at the top of frame");
});

check("an oblique view keeps the vertical up", () => {
  const oblique = cameraPlacementAtHeight(DOMAIN, UNIT, 2000, 30, 0);
  if (!oblique) return;
  assertTrue(oblique.viewUp[2] === 1, "the vertical stays up away from the pole");
});

// ---------- Azimuth ----------

check("the requested bearing is the bearing produced", () => {
  for (const azimuth of [0, Math.PI / 2, -Math.PI / 3, 2.4]) {
    const placement = cameraPlacementAtHeight(DOMAIN, UNIT, 1000, 20, azimuth);
    if (!placement) continue;
    const measured = cameraAzimuthRadians(placement.position, placement.focalPoint);
    const delta = Math.abs(Math.atan2(Math.sin(measured - azimuth), Math.cos(measured - azimuth)));
    assertTrue(delta < 1e-9, `bearing ${azimuth} round-trips (got ${measured})`);
  }
});

check("a camera directly over its target keeps the opening bearing", () => {
  const azimuth = cameraAzimuthRadians([10, 20, 5000], [10, 20, 0]);
  assertTrue(azimuth === DEFAULT_AZIMUTH_RADIANS, "an undefined bearing does not collapse to due east");
});

check("changing the look angle preserves the bearing", () => {
  const azimuth = 1.1;
  const low = cameraPlacementAtHeight(DOMAIN, UNIT, 1000, 15, azimuth);
  const oblique = cameraPlacementAtHeight(DOMAIN, UNIT, 1000, 30, azimuth);
  if (!low || !oblique) return;
  const a = cameraAzimuthRadians(low.position, low.focalPoint);
  const b = cameraAzimuthRadians(oblique.position, oblique.focalPoint);
  assertTrue(Math.abs(a - b) < 1e-9, "the orbit does not spin when the pitch changes");
});

check("every compass preset produces the requested viewing bearing", () => {
  for (const preset of CAMERA_BEARING_PRESETS) {
    const placement = cameraPlacementAtHeight(
      DOMAIN,
      UNIT,
      1000,
      20,
      cameraPositionAzimuthForBearing(preset.degrees),
    );
    if (!placement) continue;
    const viewX = placement.focalPoint[0] - placement.position[0];
    const viewY = placement.focalPoint[1] - placement.position[1];
    const bearing = ((Math.atan2(viewX, viewY) * 180 / Math.PI) + 360) % 360;
    assertTrue(
      Math.abs(bearing - preset.degrees) < 1e-9,
      `${preset.label} looks toward ${preset.degrees} degrees (got ${bearing})`,
    );
  }
});

// ---------- Refusals ----------

check("an unusable domain or scale yields no placement", () => {
  assertTrue(cameraPlacementAtHeight(null, UNIT, 0, 0, 0) === null, "null bounds");
  assertTrue(
    cameraPlacementAtHeight(DOMAIN, { xy: Number.NaN, z: 1 }, 0, 0, 0) === null,
    "NaN horizontal scale",
  );
  assertTrue(
    cameraPlacementAtHeight(DOMAIN, { xy: 0, z: 1 }, 0, 0, 0) === null,
    "a zero horizontal scale would collapse the domain to a point",
  );
  assertTrue(
    cameraPlacementAtHeight(DOMAIN, UNIT, 0, 0, Number.NaN) === null,
    "NaN bearing",
  );
  assertTrue(
    cameraPlacementAtHeight(
      { lower: [0, 0, Number.NaN], upper: DOMAIN.upper },
      UNIT,
      0,
      0,
      0,
    ) === null,
    "a non-finite corner",
  );
});

check("a degenerate domain still puts the camera off its focal point", () => {
  const point: DomainBoundsM = { lower: [5, 5, 0], upper: [5, 5, 100] };
  const placement = cameraPlacementAtHeight(point, UNIT, 50, 30, 0);
  if (!placement) return;
  assertTrue(
    magnitude(placement.position, placement.focalPoint) > 0,
    "a zero-width domain does not render from inside itself",
  );
});

check("every placement component is finite", () => {
  for (const preset of ELEVATION_PRESETS) {
    for (const heightM of [0, 18000]) {
      const placement = cameraPlacementAtHeight(DOMAIN, { xy: 2, z: 6 }, heightM, preset.degrees, 2);
      if (!placement) continue;
      const all = [...placement.position, ...placement.focalPoint, ...placement.viewUp];
      assertTrue(
        all.every((value) => Number.isFinite(value)),
        `${preset.id} at ${heightM} m is finite`,
      );
    }
  }
});

// ---------- Framing distance ----------

/*
 * The first cut sized the eye at 1.4x the wider horizontal span, which stands
 * roughly twice too close for a domain of this shape and pushed the domain box
 * off all four edges. Distance has to come from the bounding sphere and the
 * field of view, the way the scenes' own reset does.
 */
check("the fallback distance fits the whole domain in the frame", () => {
  const aspect = 16 / 9;
  const distance = fitDistanceM(DOMAIN, UNIT, aspect);
  assertTrue(distance != null, "a finite domain yields a distance");
  if (distance == null) return;

  const spanX = DOMAIN.upper[0] - DOMAIN.lower[0];
  const spanY = DOMAIN.upper[1] - DOMAIN.lower[1];
  const spanZ = DOMAIN.upper[2] - DOMAIN.lower[2];
  const radius = 0.5 * Math.hypot(spanX, spanY, spanZ);
  const vertical = (CAMERA_VIEW_ANGLE_DEGREES * Math.PI) / 180;
  const horizontal = 2 * Math.atan(Math.tan(vertical / 2) * aspect);
  const binding = Math.min(vertical, horizontal);
  // Every point of the domain must fall inside the narrower field of view.
  assertTrue(
    Math.asin(Math.min(1, radius / distance)) <= binding / 2 + 1e-9,
    "the bounding sphere subtends no more than the binding field of view",
  );
  assertTrue(distance > 1.4 * Math.max(spanX, spanY), "it stands back further than the old heuristic");
});

check("a taller viewport is fitted on its narrower horizontal field of view", () => {
  const wide = fitDistanceM(DOMAIN, UNIT, 2);
  const tall = fitDistanceM(DOMAIN, UNIT, 0.5);
  assertTrue(wide != null && tall != null, "both aspects yield a distance");
  if (wide == null || tall == null) return;
  assertTrue(tall > wide, "the narrow viewport stands further back");
});

check("vertical exaggeration pushes the eye back", () => {
  const trueScale = fitDistanceM(DOMAIN, UNIT, 1.5);
  const exaggerated = fitDistanceM(DOMAIN, { xy: 1, z: 6 }, 1.5);
  assertTrue(trueScale != null && exaggerated != null, "both scales yield a distance");
  if (trueScale == null || exaggerated == null) return;
  // Six times the height is six times the vertical span; a distance blind to z
  // would return the same number and clip the domain top and bottom.
  assertTrue(exaggerated > trueScale, "a six-times-exaggerated domain is framed from further away");
});

check("a degenerate aspect or domain refuses rather than returning a bad distance", () => {
  assertTrue(fitDistanceM(null, UNIT, 1.5) == null, "no bounds");
  assertTrue(fitDistanceM(DOMAIN, UNIT, 0) == null, "zero aspect");
  assertTrue(fitDistanceM(DOMAIN, UNIT, Number.NaN) == null, "non-finite aspect");
  assertTrue(
    fitDistanceM({ lower: [0, 0, 0], upper: [0, 0, 0] }, UNIT, 1.5) == null,
    "a domain with no extent",
  );
});

/*
 * The point of the feature: descend to a low-level feature without losing the
 * zoom you already framed it at. Recomputing the distance would pull the eye
 * back out to the whole domain every time the slider moved.
 */
check("a height change keeps the distance the camera already stood at", () => {
  const kept = 12345;
  for (const heightM of [0, 500, 6000]) {
    const placement = cameraPlacementAtHeight(DOMAIN, UNIT, heightM, 15, 0.4, { distanceM: kept });
    assertTrue(placement != null, `placement exists at ${heightM} m`);
    if (!placement) continue;
    assertTrue(
      Math.abs(magnitude(placement.position, placement.focalPoint) - kept) < 1e-6,
      `the eye stays ${kept} m out at ${heightM} m`,
    );
  }
});

check("an orientation change cannot stand closer than the domain fit", () => {
  const aspect = 1.5;
  const fitted = fitDistanceM(DOMAIN, UNIT, aspect);
  assertTrue(fitted != null, "the orientation fit exists");
  if (fitted == null) return;
  const placement = cameraPlacementAtHeight(DOMAIN, UNIT, 3000, 30, 0.4, {
    distanceM: fitted * 0.5,
    aspect,
    minimumFitDistance: true,
  });
  assertTrue(placement != null, "the orientation placement exists");
  if (!placement) return;
  assertTrue(
    Math.abs(magnitude(placement.position, placement.focalPoint) - fitted) < 1e-6,
    "orientation controls retain a complete domain frame",
  );
});

check("an unusable current distance falls back to the domain fit", () => {
  const aspect = 1.5;
  const fitted = fitDistanceM(DOMAIN, UNIT, aspect);
  assertTrue(fitted != null, "the fallback exists");
  for (const distanceM of [null, 0, -10, Number.NaN, Number.POSITIVE_INFINITY]) {
    const placement = cameraPlacementAtHeight(DOMAIN, UNIT, 3000, 30, 0.4, { distanceM, aspect });
    assertTrue(placement != null, `placement exists for ${String(distanceM)}`);
    if (!placement || fitted == null) continue;
    assertTrue(
      Math.abs(magnitude(placement.position, placement.focalPoint) - fitted) < 1e-6,
      `${String(distanceM)} falls back to the fitted distance`,
    );
  }
});

// ---------- Slider resolution ----------

check("the height step resolves storm-scale features", () => {
  assertTrue(CAMERA_HEIGHT_STEP_M > 0, "the step is positive");
  // A low-level mesocyclone is diagnosed over the lowest kilometre, so the
  // slider has to offer several stops inside it.
  assertTrue(1000 / CAMERA_HEIGHT_STEP_M >= 10, "at least ten stops in the lowest kilometre");
});

// ---------- Report ----------

console.log(`\ncamera elevation tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  " + f);
  process.exit(1);
}
process.exit(0);
