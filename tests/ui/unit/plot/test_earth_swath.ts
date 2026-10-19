/*
 * Earth-coordinate swath accumulation (docs/domain.md).
 *
 * Accumulating a lifetime swath by grid index is correct only while the grid is
 * fixed to the Earth. Once the domain translates, cell (i, j) is a different
 * place in every frame and the swath smears along the direction of motion.
 */

import {
  describeEarthSwath,
  earthCellOffset,
  earthSwathGrid,
  SWATH_3D_SOURCE_CELL_CAP,
  swathAccumulatorBudgetError,
  swathFrameGeometry,
  swathSpansMultipleFootprints,
  validGridCellCount,
  type SwathFrameGeometry,
} from "../../../../ui/src/lib/plot/earthSwath.ts";
import { resizeSwathAccumulator } from "../../../../ui/src/lib/plot/swathAccumulatorProtocol.ts";

let passed = 0;
const failures: string[] = [];
const check = (condition: boolean, label: string): void => {
  if (condition) passed++;
  else failures.push(`FAIL ${label}`);
};

const frame = (
  x: number,
  y: number,
  over: Partial<SwathFrameGeometry> = {},
): SwathFrameGeometry => ({
  nx: 100, ny: 100, dxM: 500, dyM: 500, earthOriginM: [x, y], ...over,
});

/* ---------- Stationary: the common case must stay an exact index copy ------- */

const still = earthSwathGrid([frame(0, 0), frame(0, 0), frame(0, 0)]);
check(still != null, "a stationary run yields an accumulator");
check(still!.nx === 100 && still!.ny === 100, "a stationary accumulator is exactly one footprint");
check(
  earthCellOffset(frame(0, 0), still!).join(",") === "0,0",
  "a stationary frame lands at the origin with no offset",
);
check(
  !swathSpansMultipleFootprints([frame(0, 0)], still!),
  "a stationary run is not reported as moving",
);
check(
  describeEarthSwath(still!, false).includes("fixed grid"),
  "a stationary swath discloses that it used the run's own grid",
);

// A run that declares no Earth origin at all behaves identically.
const undeclared = earthSwathGrid([
  { nx: 100, ny: 100, dxM: 500, dyM: 500 },
  { nx: 100, ny: 100, dxM: 500, dyM: 500 },
]);
check(undeclared != null && undeclared.nx === 100, "an undeclared origin is one shared footprint");

/* ---------- Translating: the accumulator must cover the union -------------- */

// 100 cells x 500 m = 49,500 m of extent; a 20,000 m eastward shift adds 40 cells.
const moved = earthSwathGrid([frame(0, 0), frame(20_000, 0)]);
check(moved != null, "a translating run yields an accumulator");
check(moved!.nx === 140, "the accumulator spans the union along the direction of motion");
check(moved!.ny === 100, "an axis with no motion is not widened");
check(moved!.originM[0] === 0, "the accumulator anchors on the earliest origin");
check(
  earthCellOffset(frame(20_000, 0), moved!).join(",") === "40,0",
  "a later frame is placed at its own Earth offset",
);
check(
  swathSpansMultipleFootprints([frame(0, 0), frame(20_000, 0)], moved!),
  "a widened accumulator is reported as moving",
);

// Motion in both axes, and a NEGATIVE origin: the accumulator must extend
// backwards rather than clipping the earlier part of the track.
const diagonal = earthSwathGrid([frame(0, 0), frame(-10_000, 5_000)]);
check(diagonal!.originM[0] === -10_000, "the accumulator extends behind a westward move");
check(diagonal!.nx === 120, "a westward move widens by its own displacement");
check(diagonal!.ny === 110, "a northward move widens the cross-track axis");
check(
  earthCellOffset(frame(0, 0), diagonal!).join(",") === "20,0",
  "the earliest frame is offset when a later frame extends behind it",
);

/* ---------- Nearest-cell binning ---------- */

// Placement rounds, so a reported extremum is always a value the model produced.
// Interpolating a MAXIMUM is not convex — it can only lower a peak — so a
// half-cell position error is the correct trade for an exact value.
const offGrid = earthSwathGrid([frame(0, 0), frame(20_240, 0)]);
check(
  earthCellOffset(frame(20_240, 0), offGrid!).join(",") === "40,0",
  "an origin between cells binds to the nearest cell, never interpolates",
);
const halfway = earthSwathGrid([frame(0, 0), frame(20_250, 0)]);
check(
  earthCellOffset(frame(20_250, 0), halfway!)[0] === 41,
  "an exactly-half-cell origin rounds consistently rather than truncating",
);
check(
  Math.abs(offGrid!.maxPlacementErrorM - 0.5 * Math.hypot(500, 500)) < 1e-9,
  "the binning error is reported as half a cell diagonal",
);
check(
  describeEarthSwath(offGrid!, true).includes("values exact"),
  "a moved swath discloses that its values are exact and positions binned",
);
check(
  describeEarthSwath(offGrid!, true).includes("Earth coordinates"),
  "a moved swath names the frame it accumulated in",
);

const originalAccumulator = new Float64Array([1, 2, 3, 4]);
const originalSeen = new Uint8Array([1, 1, 1, 1]);
const expanded = resizeSwathAccumulator(
  originalAccumulator, originalSeen, 2, 2, 4, 3, 1, 1, true,
);
check(
  [expanded.accum[5], expanded.accum[6], expanded.accum[9], expanded.accum[10]].join(",") === "1,2,3,4",
  "expanding a moving swath shifts every retained value into the new Earth-grid origin",
);
check(
  expanded.seen.reduce((sum, value) => sum + value, 0) === 4,
  "expanding a swath preserves exact seen-cell coverage without fabricating values",
);
let invalidResize = "";
try {
  resizeSwathAccumulator(originalAccumulator, originalSeen, 2, 2, 2, 2, 1, 0, true);
} catch (error) {
  invalidResize = error instanceof Error ? error.message : String(error);
}
check(
  invalidResize.includes("does not contain"),
  "a resize that would clip prior Earth-grid values fails closed",
);

/* ---------- Manifest face origin + halo conversion ---------- */

const haloFrame = swathFrameGeometry({
  nr: 102,
  nth: 84,
  drM: 500,
  dyM: 250,
  scalarOriginM: [10_250, 20_125, 50],
  validIndexRange: { lower: [1, 2, 0], upper: [100, 81, 9] },
}, [10_000, 20_000]);
check(haloFrame != null, "manifest grid resolves to swath geometry");
check(haloFrame!.nx === 100 && haloFrame!.ny === 80, "halo cells are excluded from the footprint");
check(
  haloFrame!.earthOriginM?.join(",") === "10750,20625",
  "moving grid face origin advances to the first valid scalar centre",
);

const alreadyCropped = swathFrameGeometry({
  nr: 100,
  nth: 80,
  drM: 500,
  dyM: 250,
  scalarOriginM: [10_750, 20_625, 50],
  validIndexRange: { lower: [0, 0, 0], upper: [99, 79, 9] },
}, null, [100, 80]);
check(
  alreadyCropped!.earthOriginM?.join(",") === "10750,20625",
  "cropped-grid placement uses its shifted scalar-centre origin without a second halo shift",
);

/* ---------- Refusals ---------- */

check(earthSwathGrid([]) === null, "no frames yields no accumulator");
check(
  earthSwathGrid([frame(0, 0), frame(0, 0, { dxM: 250 })]) === null,
  "a spacing change is refused — that is a regrid, not a binning problem",
);
check(
  earthSwathGrid([frame(0, 0), frame(0, 0, { nx: 200 })]) === null,
  "a cell-count change is refused",
);
check(
  earthSwathGrid([frame(0, 0), frame(Number.NaN, 0)]) === null,
  "a non-finite origin is refused rather than binned somewhere arbitrary",
);
check(
  earthSwathGrid([frame(0, 0, { nx: 0 })]) === null,
  "a degenerate footprint is refused",
);
check(
  earthSwathGrid([frame(0, 0, { dxM: 0 })]) === null,
  "zero spacing is refused rather than dividing by it",
);

/* ---------- Large-domain admission ---------- */

check(swathAccumulatorBudgetError(2048, 2048) === null, "a 2048-square swath fits");
check(
  swathAccumulatorBudgetError(4096, 4096)?.includes("interactive limit") === true,
  "an oversized accumulator is refused before allocation",
);
check(
  validGridCellCount({
    nr: 514,
    nth: 514,
    validIndexRange: { lower: [1, 1, 1], upper: [512, 512, 160] },
  }) === 512 * 512 * 160,
  "3D source admission counts only the halo-free domain",
);
check(
  512 * 512 * 160 < SWATH_3D_SOURCE_CELL_CAP
    && 1024 * 1024 * 256 > SWATH_3D_SOURCE_CELL_CAP,
  "beast-class exact swaths fit while gigacell sources require a column product",
);

console.log(`\nearthSwath tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length > 0) {
  console.log(failures.join("\n"));
  process.exit(1);
}
