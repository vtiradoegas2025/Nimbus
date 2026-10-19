/*
 * Signed-pair orientation for `src/lib/plot/vortexCore.ts`.
 *
 * The property that matters: the EMPHASISED surface is the feature the layer is
 * named for. A vortex core is positive vertical vorticity but NEGATIVE
 * perturbation pressure, so one shared sign convention draws a pressure high
 * where the user asked for the well.
 *
 * Run via:
 *   cd ui && npm run test:vortex-core
 */

import {
  assessMesocycloneEnvelope,
  assessMesocycloneContext,
  assessTornadoScaleVortex,
  formatCoreThreshold,
  mesocycloneEnvelopeThreshold,
  VORTEX_CORE_FIELDS,
  vortexCoreContours,
  vortexCoreDescription,
  vortexCoreLabel,
  vortexCoreReferenceMagnitude,
  stableVortexReferenceMagnitude,
  vortexCoreSign,
} from "../../../../ui/src/lib/plot/vortexCore.ts";
import { vortexExtractionInputMode } from "../../../../ui/src/lib/vtk/vortexContourProtocol.ts";
import {
  connectedSurfaceHierarchyNearestAnchor,
  connectedSurfaceNearestAnchor,
} from "../../../../ui/src/lib/vtk/vortexSurfaceConnectivity.ts";
import { deriveTornadicCondensation } from "../../../../ui/src/lib/plot/tornadicCondensation.ts";

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

// ---------- Sign ----------

check("vertical vorticity cores on the positive side", () => {
  assertTrue(vortexCoreSign("vorticity_z") === 1, "vorticity_z is +1");
});

check("perturbation pressure cores on the negative side", () => {
  // The defect this guards: p' was contoured at +threshold as the emphasised
  // surface, which is a pressure EXCESS, while the well was drawn faintly.
  assertTrue(vortexCoreSign("p_prime") === -1, "p_prime is -1");
});

check("an unknown or absent field falls back to the rotation convention", () => {
  assertTrue(vortexCoreSign(null) === 1, "null");
  assertTrue(vortexCoreSign(undefined) === 1, "undefined");
  assertTrue(vortexCoreSign("") === 1, "empty");
  assertTrue(vortexCoreSign("vorticity_magnitude") === 1, "unknown field");
});

check("compact cores use the selected sign's peak instead of a domain-wide percentile", () => {
  const stats = { min: -800, max: 0.12, p2: -3, p98: 0.004 };
  assertTrue(
    vortexCoreReferenceMagnitude("vorticity_z", stats) === 0.12,
    "cyclonic core uses positive maximum",
  );
  assertTrue(
    vortexCoreReferenceMagnitude("p_prime", stats) === 800,
    "pressure well uses negative minimum magnitude",
  );
});

// ---------- Contours ----------

check("the two contours are opposite in sign and equal in magnitude", () => {
  for (const field of VORTEX_CORE_FIELDS) {
    const contours = vortexCoreContours(field, 0.02);
    assertTrue(contours != null, `${field} yields contours`);
    if (!contours) continue;
    assertTrue(contours.core === -contours.opposite, `${field} is a mirrored pair`);
    assertTrue(
      Math.abs(contours.core) === 0.02 && Math.abs(contours.opposite) === 0.02,
      `${field} keeps the threshold magnitude`,
    );
  }
});

check("playback holds one dimensional contour reference across changing frame peaks", () => {
  assertTrue(
    stableVortexReferenceMagnitude(0.18, 0.12, true) === 0.12,
    "playing retains the settled reference",
  );
  assertTrue(
    stableVortexReferenceMagnitude(0.18, 0.12, false) === 0.18,
    "paused inspection follows the selected frame",
  );
  assertTrue(
    stableVortexReferenceMagnitude(null, 0.12, false) === 0.12,
    "a preparation gap does not collapse the contour",
  );
});

check("transient preparation gaps keep the contour worker warm", () => {
  assertTrue(vortexExtractionInputMode(true, false, true) === "hold", "enabled gap holds");
  assertTrue(vortexExtractionInputMode(true, true, true) === "extract", "prepared input extracts");
  assertTrue(vortexExtractionInputMode(false, false, true) === "stop", "disabled layer stops");
  assertTrue(vortexExtractionInputMode(true, true, false) === "stop", "no contours stops");
});

check("the emphasised contour is positive for vorticity and negative for pressure", () => {
  assertTrue(vortexCoreContours("vorticity_z", 0.02)!.core > 0, "zeta core is positive");
  assertTrue(vortexCoreContours("p_prime", 400)!.core < 0, "p' core is negative");
  assertTrue(vortexCoreContours("p_prime", 400)!.opposite > 0, "p' opposite is positive");
});

check("a threshold that cannot define a surface is refused", () => {
  for (const bad of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
    assertTrue(vortexCoreContours("vorticity_z", bad) == null, `refuses ${String(bad)}`);
  }
});

check("the mesocyclone envelope uses dimensional vorticity and never pressure", () => {
  assertTrue(mesocycloneEnvelopeThreshold("vorticity_z", 0.08) === 0.02, "quarter-core contour");
  assertTrue(mesocycloneEnvelopeThreshold("vorticity_z", 0.03) === 0.01, "physical floor");
  assertTrue(mesocycloneEnvelopeThreshold("vorticity_z", 0.011) == null, "no indistinct envelope");
  assertTrue(mesocycloneEnvelopeThreshold("p_prime", 400) == null, "pressure cannot prove rotation");
});

check("a mesocyclone-scale surface must enclose the core and be resolved in depth and width", () => {
  const core: [number, number, number, number, number, number] = [4_000, 5_000, 4_000, 5_000, 0, 1_500];
  const present = assessMesocycloneEnvelope(
    [1_000, 8_000, 1_500, 7_500, 0, 5_000],
    core,
    [250, 250, 250],
  );
  assertTrue(present.present && present.reason === "present", "deep enclosing circulation");
  assertTrue(
    assessMesocycloneEnvelope([8_000, 12_000, 8_000, 12_000, 0, 5_000], core, [250, 250, 250]).reason === "detached",
    "detached rotation rejected",
  );
  assertTrue(
    assessMesocycloneEnvelope([1_000, 8_000, 1_500, 7_500, 0, 1_000], core, [250, 250, 250]).reason === "too-shallow",
    "shallow circulation rejected",
  );
  assertTrue(
    assessMesocycloneEnvelope([4_000, 5_000, 4_000, 5_000, 0, 5_000], core, [100, 100, 250]).reason === "too-narrow",
    "narrow column is not mislabeled as mesocyclone-scale",
  );
  assertTrue(
    assessMesocycloneEnvelope(
      [100, 7_900, 1_500, 7_500, 0, 5_000],
      core,
      [100, 100, 100],
      [0, 8_000, 0, 8_000, 0, 6_000],
    ).reason === "boundary-clipped",
    "an envelope clipped by the retained ROI is withheld",
  );
});

check("meter-scale tornado detail does not impersonate mesoscale context", () => {
  const tornadoOnly = assessMesocycloneContext([200, 200, 200], [1, 1, 1]);
  assertTrue(!tornadoOnly.qualified && tornadoOnly.reason === "too-narrow",
    "a 200 m native brick skips impossible mesocyclone extraction");
  const multiscale = assessMesocycloneContext([224, 224, 160], [30, 30, 30]);
  assertTrue(multiscale.qualified, "a 30 m multi-kilometre brick can classify a mesocyclone envelope");
});

check("a tornado-scale label requires a resolved near-ground compact vertical core", () => {
  const core: [number, number, number, number, number, number] = [
    4_000, 5_200, 4_000, 5_100, 50, 1_050,
  ];
  assertTrue(
    assessTornadoScaleVortex(core, [100, 100, 100], 50).present,
    "100 m near-ground core qualifies",
  );
  assertTrue(
    assessTornadoScaleVortex(core, [500, 500, 250], 50).reason === "under-resolved",
    "coarse grids do not acquire a tornado label",
  );
  assertTrue(
    assessTornadoScaleVortex([4_000, 5_200, 4_000, 5_100, 2_000, 3_000], [100, 100, 100], 50).reason === "aloft",
    "an elevated core remains a low-level-vortex candidate",
  );
});

check("vortex connectivity follows the solved anchor instead of the largest unrelated blob", () => {
  const points = Float32Array.from([
    0, 0, 0, 1, 0, 0, 0, 1, 0,
    100, 100, 0, 102, 100, 0, 100, 102, 0,
  ]);
  const polys = Uint32Array.from([3, 0, 1, 2, 3, 3, 4, 5]);
  const selected = connectedSurfaceNearestAnchor({
    points,
    polys,
    normals: new Float32Array(points.length),
    bounds: [0, 102, 0, 102, 0, 0],
    triangles: 2,
    bytes: points.byteLength + polys.byteLength,
  }, [101, 101, 0]);
  assertTrue(selected?.triangles === 1, "one connected component retained");
  assertTrue(
    selected != null && selected.points.every((value, index) => index % 3 === 2 || value >= 100),
    "the component nearest the physical anchor is retained",
  );
});

check("high-resolution vortex connectivity retains nearby resolved companion cores", () => {
  const points = Float32Array.from([
    0, 0, 0, 1, 0, 0, 0, 1, 0,
    30, 0, 0, 31, 0, 0, 30, 1, 0,
    2_000, 0, 0, 2_001, 0, 0, 2_000, 1, 0,
  ]);
  const polys = Uint32Array.from([
    3, 0, 1, 2,
    3, 3, 4, 5,
    3, 6, 7, 8,
  ]);
  const hierarchy = connectedSurfaceHierarchyNearestAnchor({
    points,
    polys,
    normals: new Float32Array(points.length),
    bounds: [0, 2_001, 0, 1, 0, 0],
    triangles: 3,
    bytes: points.byteLength + polys.byteLength,
  }, [0, 0, 0], 100, { maxComponents: 8, minimumTriangles: 1 });
  assertTrue(hierarchy.componentCount === 2, "two nearby suction-vortex components retained");
  assertTrue(hierarchy.primary?.triangles === 1, "physical classification remains tied to one primary core");
  assertTrue(hierarchy.combined?.triangles === 2, "companion cores share one bounded GPU draw buffer");
  assertTrue(
    hierarchy.combined != null && Math.max(...hierarchy.combined.points) < 100,
    "a remote same-threshold blob is excluded from the local hierarchy",
  );
});

check("tornadic condensation is solved-field evidence, not a procedural funnel", () => {
  const shape: [number, number, number] = [51, 51, 20];
  const spacing: [number, number, number] = [100, 100, 100];
  const origin: [number, number, number] = [0, 0, 0];
  const size = shape[0] * shape[1] * shape[2];
  const tuple = (x: number, y: number, z: number) =>
    x + shape[0] * (y + shape[1] * z);
  const vorticity = new Float32Array(size);
  const wallCloud = new Float32Array(size);
  for (let z = 1; z <= 10; z++) {
    for (let y = 24; y <= 26; y++) {
      for (let x = 24; x <= 26; x++) vorticity[tuple(x, y, z)] = 0.2;
    }
  }
  // A solved environmental cloud deck whose base is 1.2 km everywhere.
  for (let z = 12; z <= 18; z++) {
    for (let y = 0; y < shape[1]; y++) {
      for (let x = 0; x < shape[0]; x++) wallCloud[tuple(x, y, z)] = 0.001;
    }
  }
  const base = { shape, spacing, origin };
  const wall = deriveTornadicCondensation(
    { ...base, values: wallCloud },
    { ...base, values: vorticity },
    [2_500, 2_500, 100],
    [2_400, 2_600, 2_400, 2_600, 100, 1_000],
    0.1,
    0.00005,
    true,
  );
  assertTrue(wall.assessment.visible, "rotating solved cloud can be displayed");
  assertTrue(!wall.assessment.present && wall.assessment.reason === "rotating-cloud",
    "an ordinary wall-cloud base is not promoted to a funnel");
  assertTrue(wall.assessment.downwardExtensionM === 0,
    "no descent below the surrounding cloud base");

  const funnelCloud = wallCloud.slice();
  // Actual condensate descends six model levels below the environmental deck.
  for (let z = 6; z < 12; z++) {
    for (let y = 24; y <= 26; y++) {
      for (let x = 24; x <= 26; x++) funnelCloud[tuple(x, y, z)] = 0.001;
    }
  }
  const funnel = deriveTornadicCondensation(
    { ...base, values: funnelCloud },
    { ...base, values: vorticity },
    [2_500, 2_500, 100],
    [2_400, 2_600, 2_400, 2_600, 100, 1_000],
    0.1,
    0.00005,
    true,
  );
  assertTrue(funnel.assessment.present && funnel.assessment.reason === "resolved-funnel",
    "a vertically connected solved extension earns the funnel label");
  assertTrue(funnel.assessment.downwardExtensionM === 600,
    "extension is measured against the local environmental cloud base");
  assertTrue(funnel.image?.values[tuple(25, 25, 6)]! > 0,
    "the returned volume retains actual solved condensate");
  assertTrue(funnel.image?.values[tuple(0, 0, 12)] === 0,
    "unrelated environmental cloud is excluded from the diagnostic volume");

  const unqualified = deriveTornadicCondensation(
    { ...base, values: funnelCloud },
    { ...base, values: vorticity },
    [2_500, 2_500, 100],
    [2_400, 2_600, 2_400, 2_600, 100, 1_000],
    0.1,
    0.00005,
    false,
  );
  assertTrue(!unqualified.assessment.present
    && unqualified.assessment.reason === "no-qualified-vortex",
  "condensation cannot make an unqualified circulation tornadic");
});

// ---------- Labels ----------

check("a label states the inequality the surface actually draws", () => {
  assertTrue(vortexCoreLabel(0.02) === ">= +0.02", "positive contour");
  assertTrue(vortexCoreLabel(-400) === "<= -400", "negative contour");
});

check("the pressure legend never labels a surface with a rotation direction", () => {
  // "anticyclonic" names a rotation that a pressure field cannot show.
  const core = vortexCoreDescription("p_prime", "core");
  const opposite = vortexCoreDescription("p_prime", "opposite");
  assertTrue(core === "pressure deficit", `core reads "${core}"`);
  assertTrue(opposite === "pressure excess", `opposite reads "${opposite}"`);
  assertTrue(
    !/cyclonic/.test(core) && !/cyclonic/.test(opposite),
    "no rotation wording on a pressure field",
  );
});

check("the rotation legend keeps the cyclonic wording", () => {
  assertTrue(vortexCoreDescription("vorticity_z", "core") === "cyclonic", "core");
  assertTrue(vortexCoreDescription("vorticity_z", "opposite") === "anticyclonic", "opposite");
});

check("the emphasised label and the emphasised contour agree in sign", () => {
  for (const [field, threshold] of [["vorticity_z", 0.02], ["p_prime", 400]] as const) {
    const contours = vortexCoreContours(field, threshold)!;
    const label = vortexCoreLabel(contours.core);
    assertTrue(
      contours.core >= 0 ? label.startsWith(">= +") : label.startsWith("<= -"),
      `${field} label matches its contour sign`,
    );
  }
});

// ---------- Threshold formatting ----------

check("the threshold readout stays compact across both field scales", () => {
  // zeta is ~1e-2, p' is ~1e2; one formatter has to read well for both.
  assertTrue(formatCoreThreshold(0.0123) === "0.0123", "zeta magnitude");
  assertTrue(formatCoreThreshold(412.7) === "413", "pressure magnitude");
  assertTrue(formatCoreThreshold(0.000123).includes("e"), "very small goes exponential");
  assertTrue(formatCoreThreshold(1.5e6).includes("e"), "very large goes exponential");
  assertTrue(formatCoreThreshold(0) === "0", "zero stays plain");
});

console.log(`\nvortex core tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  " + f);
  process.exit(1);
}
process.exit(0);
