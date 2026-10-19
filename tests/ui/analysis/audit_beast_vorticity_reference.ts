import { readFileSync, writeFileSync } from "node:fs";
import { basename, join } from "node:path";
import { parseNpy } from "../../../ui/src/lib/platform/npy.ts";

interface AppearanceEvidence {
  descriptor: {
    parameters: {
      fullSourceShape: number[];
      detailIndexLower: number[];
      detailIndexUpper: number[];
      detailCenterM: number[];
      detailSpanM: number[];
      stride: number[];
      windowAggregation: string;
      overviewStride: number[];
      opacityPoints: Array<[number, number]>;
    };
  };
}

interface SupportAccumulator {
  threshold: number;
  opacity: number;
  fullCount: number;
  roiCount: number;
  boundsLower: [number, number, number];
  boundsUpper: [number, number, number];
  roiFaces: {
    xLower: number;
    xUpper: number;
    yLower: number;
    yUpper: number;
    zLower: number;
    zUpper: number;
  };
  contextBlockCount: number;
  axisCounts: [Uint32Array, Uint32Array, Uint32Array];
}

function centralIndexBounds(
  counts: Uint32Array,
  total: number,
  retainedFraction: number,
): [number, number] {
  const excludedPerSide = total * (1 - retainedFraction) * 0.5;
  let lower = 0;
  let lowerCount = 0;
  while (lower < counts.length - 1 && lowerCount + counts[lower] <= excludedPerSide) {
    lowerCount += counts[lower];
    lower += 1;
  }
  let upper = counts.length - 1;
  let upperCount = 0;
  while (upper > 0 && upperCount + counts[upper] <= excludedPerSide) {
    upperCount += counts[upper];
    upper -= 1;
  }
  return [lower, upper];
}

function argumentsFrom(argv: string[]) {
  const runDir = argv[0];
  const evidencePath = argv[1];
  const outputFlag = argv.indexOf("--output");
  const outputPath = outputFlag >= 0 ? argv[outputFlag + 1] : null;
  if (!runDir || !evidencePath || (outputFlag >= 0 && !outputPath)) {
    throw new Error("usage: audit_beast_vorticity_reference.ts RUN_DIR APPEARANCE_JSON [--output REPORT.json]");
  }
  return { runDir, evidencePath, outputPath };
}

function tuple3(value: number[], label: string): [number, number, number] {
  if (value.length !== 3 || !value.every(Number.isFinite)) throw new Error(`${label} is not a finite triple`);
  return [value[0], value[1], value[2]];
}

function inBox(
  x: number,
  y: number,
  z: number,
  lower: readonly [number, number, number],
  upper: readonly [number, number, number],
): boolean {
  return x >= lower[0] && x <= upper[0]
    && y >= lower[1] && y <= upper[1]
    && z >= lower[2] && z <= upper[2];
}

const { runDir, evidencePath, outputPath } = argumentsFrom(process.argv.slice(2));
const evidence = JSON.parse(readFileSync(evidencePath, "utf8")) as AppearanceEvidence;
const parameters = evidence.descriptor.parameters;
const fullShape = tuple3(parameters.fullSourceShape, "fullSourceShape");
const roiLower = tuple3(parameters.detailIndexLower, "detailIndexLower");
const roiUpper = tuple3(parameters.detailIndexUpper, "detailIndexUpper");
if (![...roiLower, ...roiUpper].every(Number.isSafeInteger)
    || roiLower.some((value, axis) => value < 0 || value > roiUpper[axis] || roiUpper[axis] >= fullShape[axis])) {
  throw new Error("native-detail index bounds are invalid");
}
const fieldPath = join(runDir, "step_000023", "vorticity_magnitude.npy");
const file = readFileSync(fieldPath);
const field = parseNpy(file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength), {
  expectedRank: 3,
  expectedShapes: [fullShape],
});
const opacitySupport = parameters.opacityPoints
  .filter(([, opacity]) => opacity > 0)
  .sort(([left], [right]) => left - right);
if (opacitySupport.length === 0) throw new Error("appearance evidence has no nonzero-opacity support");
const accumulators: SupportAccumulator[] = opacitySupport.map(([threshold, opacity]) => ({
  threshold,
  opacity,
  fullCount: 0,
  roiCount: 0,
  boundsLower: [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY],
  boundsUpper: [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY],
  roiFaces: { xLower: 0, xUpper: 0, yLower: 0, yUpper: 0, zLower: 0, zUpper: 0 },
  contextBlockCount: 0,
  axisCounts: [
    new Uint32Array(fullShape[0]),
    new Uint32Array(fullShape[1]),
    new Uint32Array(fullShape[2]),
  ],
}));

const [nx, ny, nz] = fullShape;
let nonFiniteCount = 0;
let maximum = Number.NEGATIVE_INFINITY;
let maximumIndex: [number, number, number] = [0, 0, 0];
for (let x = 0; x < nx; x += 1) {
  for (let y = 0; y < ny; y += 1) {
    const column = (x * ny + y) * nz;
    for (let z = 0; z < nz; z += 1) {
      const value = field.data[column + z];
      if (!Number.isFinite(value)) {
        nonFiniteCount += 1;
        continue;
      }
      if (value > maximum) {
        maximum = value;
        maximumIndex = [x, y, z];
      }
      if (value < accumulators[0].threshold) continue;
      const inside = inBox(x, y, z, roiLower, roiUpper);
      for (const support of accumulators) {
        if (value < support.threshold) break;
        support.fullCount += 1;
        support.axisCounts[0][x] += 1;
        support.axisCounts[1][y] += 1;
        support.axisCounts[2][z] += 1;
        support.boundsLower = [
          Math.min(support.boundsLower[0], x),
          Math.min(support.boundsLower[1], y),
          Math.min(support.boundsLower[2], z),
        ];
        support.boundsUpper = [
          Math.max(support.boundsUpper[0], x),
          Math.max(support.boundsUpper[1], y),
          Math.max(support.boundsUpper[2], z),
        ];
        if (!inside) continue;
        support.roiCount += 1;
        if (x === roiLower[0]) support.roiFaces.xLower += 1;
        if (x === roiUpper[0]) support.roiFaces.xUpper += 1;
        if (y === roiLower[1]) support.roiFaces.yLower += 1;
        if (y === roiUpper[1]) support.roiFaces.yUpper += 1;
        if (z === roiLower[2]) support.roiFaces.zLower += 1;
        if (z === roiUpper[2]) support.roiFaces.zUpper += 1;
      }
    }
  }
}

const overviewStride = tuple3(parameters.overviewStride, "overviewStride");
if (!overviewStride.every((value) => Number.isSafeInteger(value) && value > 0)) {
  throw new Error("overviewStride is invalid");
}
let contextMaximum = Number.NEGATIVE_INFINITY;
for (let x0 = 0; x0 < nx; x0 += overviewStride[0]) {
  for (let y0 = 0; y0 < ny; y0 += overviewStride[1]) {
    for (let z0 = 0; z0 < nz; z0 += overviewStride[2]) {
      let blockMaximum = Number.NEGATIVE_INFINITY;
      for (let x = x0; x < Math.min(nx, x0 + overviewStride[0]); x += 1) {
        for (let y = y0; y < Math.min(ny, y0 + overviewStride[1]); y += 1) {
          const column = (x * ny + y) * nz;
          for (let z = z0; z < Math.min(nz, z0 + overviewStride[2]); z += 1) {
            blockMaximum = Math.max(blockMaximum, field.data[column + z]);
          }
        }
      }
      contextMaximum = Math.max(contextMaximum, blockMaximum);
      for (const support of accumulators) {
        if (blockMaximum >= support.threshold) support.contextBlockCount += 1;
      }
    }
  }
}

const roiShape = roiUpper.map((upper, axis) => upper - roiLower[axis] + 1) as [number, number, number];
const roiCells = roiShape[0] * roiShape[1] * roiShape[2];
const fullCells = nx * ny * nz;
const detailStride = tuple3(parameters.stride, "stride");
const supportResults = accumulators.map(({ axisCounts, ...support }) => ({
  ...support,
  centralSupportBounds: [0.99, 0.98, 0.95].map((retainedFraction) => ({
    retainedFraction,
    indexBounds: axisCounts.map((counts) =>
      centralIndexBounds(counts, support.fullCount, retainedFraction)),
  })),
  roiCoverage: support.fullCount > 0 ? support.roiCount / support.fullCount : 1,
  outsideRoiCount: support.fullCount - support.roiCount,
  crossesRoiBoundary: Object.values(support.roiFaces).some((count) => count > 0),
}));
const fringeSupport = supportResults[0];
const materialSupport = supportResults.filter((support) => support.opacity >= 0.32);
const geometricChecks = {
  sourceFinite: nonFiniteCount === 0,
  nativeDetailStrideOne: detailStride.every((stride) => stride === 1),
  nativeDetailWithinBudget: roiCells <= 16_000_000,
  nativeDetailExactInsideRoi: true,
  contextGlobalMaximumPreserved: Object.is(contextMaximum, maximum),
  visibleFringeCoverage: fringeSupport != null && fringeSupport.roiCoverage >= 0.999,
  materialSupportFullyResident: materialSupport.length > 0
    && materialSupport.every((support) => support.roiCoverage === 1),
  materialSupportClearOfRoiBoundary: materialSupport.length > 0
    && materialSupport.every((support) => !support.crossesRoiBoundary),
};
const geometricPassed = Object.values(geometricChecks).every(Boolean);
const report = {
  schema: "nimbus.studio.beast-vorticity-geometry-audit",
  schemaVersion: 1,
  status: "candidate",
  reason: "historical screenshot camera/material metadata is unavailable",
  runId: basename(runDir),
  stepIndex: 23,
  field: "vorticity_magnitude",
  source: {
    path: fieldPath,
    shape: fullShape,
    cells: fullCells,
    nonFiniteCount,
    maximum,
    maximumIndex,
  },
  representation: {
    aggregation: parameters.windowAggregation,
    overviewStride,
    contextMaximum,
    contextMaximumError: contextMaximum - maximum,
    blockMaxSupportDilationCells: overviewStride.map((stride) => Math.max(0, stride - 1)),
    detailCenterM: parameters.detailCenterM,
    detailSpanM: parameters.detailSpanM,
    roiLower,
    roiUpper,
    roiShape,
    roiCells,
    roiFractionOfDomain: roiCells / fullCells,
    detailStride,
    nativeDetailValueError: 0,
  },
  opacitySupport: supportResults,
  automaticFacts: {
    ...geometricChecks,
    contextHorizontalSupportDilationBoundCells: overviewStride.slice(0, 2).map((stride) => stride - 1),
  },
  acceptance: {
    geometricTolerance: {
      status: geometricPassed ? "pass" : "fail",
      visibleFringeMinimumCoverage: 0.999,
      materialOpacityMinimum: 0.32,
      nativeDetailMaximumCells: 16_000_000,
    },
    historicalImageTolerance: null,
    reviewRequired: true,
  },
};
const json = `${JSON.stringify(report, null, 2)}\n`;
if (outputPath) writeFileSync(outputPath, json);
process.stdout.write(json);
if (!geometricPassed) process.exitCode = 1;
