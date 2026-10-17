/*
 * Tests for src/lib/plot/domainAxes.ts — the domain box's axis rulers.
 *
 * These replace vtk.js 36's CubeAxesActor, which registers no view node and so
 * draws nothing at all. The arithmetic matters more than usual because two
 * coordinate spaces meet in it:
 *
 *   - a tick is DRAWN at its display position (true metres x the scene scale)
 *   - a tick is LABELLED with its true value
 *
 * Getting that backwards would report a storm 3.5x deeper than it is under 3.5x
 * vertical exaggeration, while looking entirely plausible. Most of what follows
 * pins that separation down.
 *
 * Run via:
 *   cd ui && npm run test:domain-axes
 */

import {
  axisTickValuesM,
  axisTickValuesIndex,
  buildDomainAxisTicks,
  chooseHorizontalCorners,
  chooseVerticalEdge,
  domainBoundsM,
  domainAxisBounds,
  describeDomainGrid,
  formatAxisTickKm,
  niceTickStepM,
  VERTICAL_EDGE_CORNERS,
  type AxisCorners,
} from "../../../../ui/src/lib/plot/domainAxes.ts";
import { cartesianScalarOrigin } from "../../../../ui/src/lib/science/cartesianInterp.ts";

let passed = 0;
let failed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed++;
  else {
    failed++;
    failures.push(`FAIL ${label}`);
  }
}

const grid = {
  nr: 61,
  nth: 61,
  nz: 22,
  drM: 100,
  dyM: 100,
  dzM: 100,
};

{
  const indices = { geometryKind: "index" as const, nr: 7, nth: 5, nz: 3 };
  equal(domainBoundsM(indices), null, "sample counts never become physical domain bounds");
  equal(domainAxisBounds(indices), { lower: [0, 0, 0], upper: [6, 4, 2] },
    "index rulers use the declared source sample indices");
  equal(describeDomainGrid(indices), { caption: "7×5×3 samples", notice: "physical spacing unavailable" },
    "the index badge reports counts and unavailable physical spacing");
  const synthetic = { ...indices, drM: 1, dyM: 1, dzM: 1, scalarOriginM: [1000, -2000, 50] as [number, number, number],
    physicalBoundsM: { lower: [1000, 2000, 50] as [number, number, number], upper: [7000, 8000, 2150] as [number, number, number] } };
  equal(domainBoundsM(synthetic), null, "synthetic metre fields cannot override index-only ownership");
  equal(describeDomainGrid(synthetic), describeDomainGrid(indices), "synthetic spacings do not claim kilometres");
  equal(domainAxisBounds(synthetic), domainAxisBounds(indices), "index coordinates do not borrow a physical origin");
  const rulers = buildDomainAxisTicks(synthetic, { xy: 2, z: 4 })!;
  equal(rulers.axisLabels.map(axis => axis.label), ["i (index)", "j (index)", "k (index)"],
    "array axes state index units without claiming Earth orientation");
  check(rulers.ticks.every(tick => Number.isInteger(tick.valueM) && tick.label === String(tick.valueM)),
    "every index tick labels an actual integral sample coordinate");
  check(rulers.ticks.filter(tick => tick.axis === "z").every(tick => tick.from[2] === tick.valueM * 4),
    "display exaggeration moves index ticks without changing their values");
  check(rulers.ticks.filter(tick => tick.axis === "x").every(tick => tick.from[0] === tick.valueM * 2),
    "horizontal display scaling does not relabel source indices");
  equal(axisTickValuesIndex(0, 1), [0, 1], "two samples do not acquire fractional index ticks");
  equal(axisTickValuesIndex(2, 6, 2), [2, 4, 6], "index tick values respect a declared sample interval");
  equal(axisTickValuesIndex(0.5, 6), [], "fractional bounds are not declared sample indices");
  equal(axisTickValuesIndex(0, 6, 0), [], "invalid tick count cannot invent index positions");
  const cropped = { ...indices, validIndexRange: { lower: [1, 1, 0] as [number, number, number], upper: [5, 3, 2] as [number, number, number] } };
  equal(domainAxisBounds(cropped), { lower: [1, 1, 0], upper: [5, 3, 2] },
    "published valid index limits retain their original source numbering");
  equal(domainAxisBounds({ ...cropped, validIndexRange: { lower: [1, 1, 0], upper: [7, 3, 2] } }), null,
    "invalid source index limits are not silently clamped");
  equal(domainAxisBounds({ ...indices, nth: undefined }), null, "missing index dimensions are not inferred from another axis");
  equal(describeDomainGrid({ ...indices, nth: undefined }),
    { caption: "i 7 · j unavailable · k 3", notice: "physical spacing unavailable" },
    "the badge identifies a missing dimension without fabricating its count");
  equal(describeDomainGrid(grid), { caption: "x 6.0 · y 6.0 · z 2.1 km", notice: null },
    "legacy physical-grid badge formatting remains unchanged");
  equal(describeDomainGrid({ ...grid, geometryKind: "physical",
    physicalBoundsM: { lower: [1000, 2000, 50], upper: [7000, 8000, 2150] } }),
  { caption: "x 6.0 · y 6.0 · z 2.1 km", notice: null },
  "declared physical bounds preserve their kilometre extent");
  equal(describeDomainGrid(null), null, "absent metadata does not produce a domain badge");
}
function equal(actual: unknown, expected: unknown, label: string): void {
  check(
    JSON.stringify(actual) === JSON.stringify(expected),
    `${label} (expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)})`,
  );
}

/* ---------- Domain bounds ---------- */
{
  const bounds = domainBoundsM(grid);
  equal(bounds?.lower, [-3000, -3000, 0], "an implicit domain is centred on x=y=0 and rises from z=0");
  equal(bounds?.upper, [3000, 3000, 2100], "an implicit domain spans (n-1) cells per axis");

  const explicit = domainBoundsM({
    ...grid,
    physicalBoundsM: { lower: [1000, 2000, 50], upper: [7000, 8000, 2150] },
  });
  equal(explicit?.lower, [1000, 2000, 50], "explicit manifest bounds win over the derived box");
  equal(explicit?.upper, [7000, 8000, 2150], "explicit manifest upper bounds are used verbatim");

  check(domainBoundsM(null) === null, "no grid yields no bounds");
  check(domainBoundsM({ ...grid, drM: 0 }) === null, "a degenerate spacing yields no bounds");
}

/* ---------- Nice tick steps ---------- */
{
  // 6000/5 = 1200, which rounds UP to the next nice mantissa rather than down:
  // 4 clean ticks beat 6 ticks on a value like 1.2 km.
  equal(niceTickStepM(6000, 5), 2000, "a 6 km span steps in round kilometres");
  equal(niceTickStepM(2100, 5), 500, "a 2.1 km span steps in half kilometres");
  equal(niceTickStepM(300, 5), 100, "a 300 m span steps in round hundreds");
  check([1, 2, 5].includes(niceTickStepM(43000, 5) / 10000), "steps stay on a 1/2/5 mantissa");
  equal(niceTickStepM(0, 5), 0, "a zero span has no step");
  equal(niceTickStepM(-10, 5), 0, "a negative span has no step");
  equal(niceTickStepM(Number.NaN, 5), 0, "a non-finite span has no step");
}

/* ---------- Tick values ---------- */
{
  equal(axisTickValuesM(0, 2100, 5), [0, 500, 1000, 1500, 2000], "ticks cover the span on round values");
  equal(axisTickValuesM(-3000, 3000, 5), [-2000, 0, 2000], "ticks span a centred axis symmetrically");

  // Repeated addition drifts; a tick that lands on 1500.0000000000002 would
  // format as a different number from its neighbours.
  const drifting = axisTickValuesM(0, 5000, 5);
  check(drifting.every((v) => Number.isInteger(v)), "accumulated float drift is rounded away");

  // A tick exactly on the domain face must survive a last-bit comparison.
  equal(axisTickValuesM(0, 2000, 4).at(-1), 2000, "a tick sitting on the upper face is kept");
  equal(axisTickValuesM(5, 5, 5), [], "a zero-width axis has no ticks");
  equal(axisTickValuesM(Number.NaN, 10, 5), [], "a non-finite bound yields no ticks");
}

/* ---------- Label formatting ---------- */
{
  equal(formatAxisTickKm(0), "0", "zero reads as a bare 0");
  equal(formatAxisTickKm(2000), "2", "a round kilometre drops its decimal");
  equal(formatAxisTickKm(500), "0.5", "a half kilometre keeps one decimal");
  equal(formatAxisTickKm(-1500), "-1.5", "negative coordinates keep their sign");
  equal(formatAxisTickKm(50), "0.05", "a sub-100 m tick keeps two decimals");
  equal(formatAxisTickKm(25000), "25", "a large value drops decimals entirely");
}

/* ---------- Edge selection follows the camera ---------- */
{
  const bounds = domainBoundsM(grid)!;
  const near = chooseHorizontalCorners(bounds, -9000, -9000);
  equal(
    [near.xAtUpperY, near.yAtUpperX],
    [false, false],
    "rulers sit on the faces nearest a camera below the domain centre",
  );
  const far = chooseHorizontalCorners(bounds, 9000, 9000);
  equal(
    [far.xAtUpperY, far.yAtUpperX],
    [true, true],
    "rulers move to the opposite faces when the camera crosses the centre",
  );

  // The height ruler must land on the box's silhouette. Both the nearest and
  // furthest corners project over the middle of the scene, straight through the
  // storm, which is what the fixed-corner first version did.
  equal(
    chooseVerticalEdge([0.62, 0.30, 0.55, 0.81]),
    VERTICAL_EDGE_CORNERS[1],
    "the height ruler takes the leftmost projected vertical edge",
  );
  equal(
    chooseVerticalEdge([null, null, 0.4, 0.9]),
    VERTICAL_EDGE_CORNERS[2],
    "unprojectable corners are skipped rather than chosen",
  );
  equal(
    chooseVerticalEdge([null, null, null, null]),
    [false, false],
    "an unprojectable box falls back to the default corner",
  );
  equal(
    chooseVerticalEdge([Number.NaN, 0.7, Number.NaN, Number.NaN]),
    VERTICAL_EDGE_CORNERS[1],
    "NaN projections are rejected as unusable",
  );
}

/* ---------- Display position vs true value ---------- */
{
  const corners: AxisCorners = { xAtUpperY: false, yAtUpperX: false, zCorner: [false, false] };
  const flat = buildDomainAxisTicks(grid, { xy: 1, z: 1 }, corners)!;
  const tall = buildDomainAxisTicks(grid, { xy: 1, z: 3.5 }, corners)!;
  check(Boolean(flat) && Boolean(tall), "ticks build at both scales");

  const zFlat = flat.ticks.filter((t) => t.axis === "z");
  const zTall = tall.ticks.filter((t) => t.axis === "z");
  equal(
    zTall.map((t) => t.label),
    zFlat.map((t) => t.label),
    "vertical exaggeration does NOT change what the height ticks say",
  );
  equal(
    zTall.map((t) => t.valueM),
    zFlat.map((t) => t.valueM),
    "height tick values stay in true metres under exaggeration",
  );
  check(
    zTall.every((t, i) => Math.abs(t.from[2] - zFlat[i].from[2] * 3.5) < 1e-6),
    "height ticks are DRAWN at the exaggerated position",
  );

  // A tick mark is a fixed-length mark, not a scaled one: building it from
  // already-scaled points is what keeps it square under exaggeration, the same
  // rule the tube plots follow (lib/vtk/vtkScaledPolyData.ts).
  const markLength = (t: { from: number[]; to: number[] }) =>
    Math.hypot(t.to[0] - t.from[0], t.to[1] - t.from[1], t.to[2] - t.from[2]);
  const lengths = tall.ticks.map(markLength);
  check(
    lengths.every((l) => Math.abs(l - lengths[0]) < 1e-3),
    "every tick mark is the same length on every axis, under exaggeration",
  );

  // The three rulers meet at a corner; three zeros stacked on one point is not
  // a reading, it is a smudge.
  check(
    !zFlat.some((t) => t.valueM === 0),
    "the height ruler drops the tick on the floor it rises from",
  );
  check(
    flat.ticks.some((t) => t.axis === "x" && t.valueM === 0),
    "the floor rulers keep their origin tick",
  );

  equal(
    flat.axisLabels.map((a) => a.label),
    ["x (km)", "y (km)", "z (km)"],
    "each axis is named with its unit",
  );

  // Corner choice must actually move the geometry, or the camera-adaptive
  // selection is decorative.
  const moved = buildDomainAxisTicks(grid, { xy: 1, z: 1 }, {
    xAtUpperY: true,
    yAtUpperX: false,
    zCorner: [true, true],
  })!;
  const xEdgeY = (t: { from: number[] }) => t.from[1];
  check(
    xEdgeY(moved.ticks.find((t) => t.axis === "x")!)
      !== xEdgeY(flat.ticks.find((t) => t.axis === "x")!),
    "flipping a corner moves the ruler to the opposite face",
  );

  check(buildDomainAxisTicks(null, { xy: 1, z: 1 }, corners) === null, "no grid yields no ticks");
  check(
    buildDomainAxisTicks(grid, { xy: Number.NaN, z: 1 }, corners) === null,
    "a non-finite scale yields no ticks rather than NaN geometry",
  );
}

/* ---------- Floor plane and volume share one frame ---------- */
{
  // The radar floor anchors to domainBoundsM; the 3-D volume anchors to
  // cartesianScalarOrigin. If those drift, the ground sheet slides out from
  // under the storm it belongs to and nothing else would catch it.
  const shape: [number, number, number] = [grid.nr, grid.nth, grid.nz];
  const origin = cartesianScalarOrigin(grid, shape);
  const bounds = domainBoundsM(grid)!;
  check(origin[0] === bounds.lower[0], "volume x origin is the floor's lower x bound");
  check(origin[1] === bounds.lower[1], "volume y origin is the floor's lower y bound");
  // z is deliberately offset: the volume's first scalar sits at a cell centre,
  // the floor is the ground plane itself.
  check(origin[2] === 0.5 * grid.dzM, "volume z origin sits half a cell above the floor");
}

console.log(`\ndomainAxes tests: ${passed} passed, ${failed} failed.`);
if (failures.length) {
  console.log(failures.join("\n"));
  process.exit(1);
}
