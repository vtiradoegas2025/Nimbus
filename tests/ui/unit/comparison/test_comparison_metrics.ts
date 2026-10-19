import {
  emptyPairMetricMoments,
  emptyVectorMetricMoments,
  finishPairMetricMoments,
  finishVectorMetricMoments,
  mergePairMetricMoments,
  mergeVectorMetricMoments,
  matchedPhysicalVolumes,
  scanPairMetricMoments,
  scanPairMetrics,
  scanVectorMetricMoments,
  scanVectorPairMetrics,
  volumeMeanPairSample,
} from "../../../../ui/src/lib/comparison/comparisonMetrics.ts";

let passed = 0;
const failures: string[] = [];
const check = (condition: boolean, label: string) => condition ? passed += 1 : failures.push(label);
const close = (left: number, right: number) => Math.abs(left - right) < 1e-10;

const scalar = await scanPairMetrics(new Float32Array([1, 3, Number.NaN]), new Float32Array([0, 1, 5]));
check(scalar.count === 2, "scalar metrics use finite pairs only");
check(close(scalar.mean, 1.5) && close(scalar.rmse, Math.sqrt(2.5)), "scalar bias and RMSE are exact");
const pairFold = emptyPairMetricMoments();
mergePairMetricMoments(pairFold, await scanPairMetricMoments(
  new Float32Array([1]), new Float32Array([0]),
));
mergePairMetricMoments(pairFold, await scanPairMetricMoments(
  new Float32Array([3, Number.NaN]), new Float32Array([1, 5]),
));
const foldedScalar = finishPairMetricMoments(pairFold);
check(
  foldedScalar.count === scalar.count && close(foldedScalar.mean, scalar.mean)
    && close(foldedScalar.rmse, scalar.rmse) && close(foldedScalar.mae, scalar.mae),
  "streamed scalar tile moments equal a whole-field exact scan",
);
const weightedScalar = finishPairMetricMoments(await scanPairMetricMoments(
  new Float32Array([10, 20]), new Float32Array([0, 0]),
  { weights: new Float64Array([1, 3]) },
));
check(close(weightedScalar.mean, 17.5) && close(weightedScalar.rmse, Math.sqrt(325)),
  "terrain scalar metrics use exact physical-volume weights");

const shape = [2, 2, 2] as const;
const leftX = new Float32Array(8).fill(2);
const leftY = new Float32Array(8).fill(4);
const rightX = new Float32Array(8).fill(1);
const rightY = new Float32Array(8).fill(2);
const full = await scanVectorPairMetrics(leftX, leftY, rightX, rightY, shape, { lower: [0, 0, 0], upper: [1, 1, 1] });
check(full.count === 8, "full vector volume scans every selected cell");
check(close(full.meanX, 1) && close(full.meanY, 2), "vector component biases are exact");
check(close(full.rmse, Math.sqrt(5)) && close(full.maxMagnitude, Math.sqrt(5)), "vector magnitude metrics are exact");
const vectorFold = emptyVectorMetricMoments();
for (const [start, end] of [[0, 3], [3, 8]] as const) {
  mergeVectorMetricMoments(vectorFold, await scanVectorMetricMoments(
    leftX.subarray(start, end), leftY.subarray(start, end),
    rightX.subarray(start, end), rightY.subarray(start, end),
  ));
}
const foldedVector = finishVectorMetricMoments(vectorFold);
check(
  foldedVector.count === full.count && close(foldedVector.meanX, full.meanX)
    && close(foldedVector.meanY, full.meanY) && close(foldedVector.rmse, full.rmse),
  "streamed vector tile moments equal a whole-field exact scan",
);
const weightedVector = finishVectorMetricMoments(await scanVectorMetricMoments(
  new Float32Array([1, 3]), new Float32Array([0, 0]),
  new Float32Array([0, 0]), new Float32Array([0, 0]),
  { weights: new Float64Array([1, 3]) },
));
check(close(weightedVector.meanX, 2.5) && close(weightedVector.rmse, Math.sqrt(7)),
  "terrain vector metrics use exact physical-volume weights");

const terrainField = (zOffset = 0) => ({
  shape: [1, 1, 2],
  data: new Float32Array([1, 2]),
  windowGeometry: {
    weighting: "physical_cell_volume" as const,
    sampleLocation: "physical_volume_centroid" as const,
    centroidsM: {
      x: new Float64Array([50, 50]), y: new Float64Array([50, 50]),
      z: new Float64Array([5 + zOffset, 20 + zOffset]),
    },
    volumesM3: new Float64Array([100, 200]),
  },
});
check(matchedPhysicalVolumes(terrainField(), terrainField())?.[1] === 200,
  "co-located terrain tiles expose their physical volumes");
let terrainMismatchRejected = false;
try { matchedPhysicalVolumes(terrainField(), terrainField(1)); } catch { terrainMismatchRejected = true; }
check(terrainMismatchRejected, "terrain metrics reject grids that are not physically co-located");
const subset = await scanVectorPairMetrics(leftX, leftY, rightX, rightY, shape, { lower: [1, 0, 0], upper: [1, 0, 1] });
check(subset.count === 2, "inclusive volumetric subregion is honored");

let rejected = false;
try {
  await scanVectorPairMetrics(leftX, leftY, rightX, rightY, shape, { lower: [1, 0, 0], upper: [0, 1, 1] });
} catch { rejected = true; }
check(rejected, "invalid volumetric bounds fail closed");

const checkerLeft = new Float32Array([0, 1, 1, 0, 1, 0, 0, 1]);
const checkerRight = new Float32Array(8).fill(0.25);
const checker = volumeMeanPairSample(
  checkerLeft, checkerRight, [2, 2, 2],
  { lower: [0, 0, 0], upper: [1, 1, 1] }, 2,
  [[0.5, 1.5], [0.5, 1.5], [0.5, 1.5]],
);
check(checker.left.length === 1, "volume display emits one block instead of a point sample");
check(close(checker.left[0], 0.5), "checkerboard display uses its block mean");
check(close(checker.right[0], 0.25) && close(checker.difference[0], 0.25), "A/B means preserve a linear difference");
check(close(checker.x[0], 1) && close(checker.y[0], 1) && close(checker.z[0], 1), "display block coordinates are volume centroids");

const stretched = volumeMeanPairSample(
  new Float32Array([0, 0, 10]), new Float32Array([0, 0, 0]), [1, 1, 3],
  { lower: [0, 0, 0], upper: [0, 0, 2] }, 3,
  [[0], [0], [0.5, 1.5, 4]],
);
check(close(stretched.left[0], 100 / 21), "stretched display weights values by physical cell width");
check(close(stretched.z[0], 2.5), "stretched display position is the physical-volume centroid");

const poisoned = volumeMeanPairSample(
  new Float32Array([1, Number.NaN]), new Float32Array([1, 2]), [1, 1, 2],
  { lower: [0, 0, 0], upper: [0, 0, 1] }, 2,
  [[0], [0], [0.5, 1.5]],
);
check(poisoned.left.length === 0, "a non-finite contributor invalidates its complete display block");

console.log(`comparison metrics: ${passed} checks passed, ${failures.length} failed`);
if (failures.length > 0) {
  failures.forEach((failure) => console.error(`FAIL ${failure}`));
  process.exit(1);
}
