import { createLocalDatasetSource } from "../../../../ui/src/lib/dataset/datasetSources.ts";
import { analyzeDatasetStructure, analyzeFieldValues, createDatasetFieldHealthReport } from "../../../../ui/src/lib/dataset/dataHealth.ts";
import type { NpyN } from "../../../../ui/src/lib/platform/npy.ts";
import type { StepInfo } from "../../../../ui/src/lib/run/runOutputs.ts";

let passed = 0; const failures: string[] = []; const check = (value: boolean, label: string) => value ? passed += 1 : failures.push(label);
const values: NpyN = { shape: [7], data: Float32Array.from([1, 2, 3, Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, 4]) };
const health = analyzeFieldValues(values);
check(health.finiteCount === 4 && health.nanCount === 1 && health.positiveInfinityCount === 1 && health.negativeInfinityCount === 1, "non-finite classes");
check(health.minimum === 1 && health.maximum === 4 && health.mean === 2.5, "finite statistics");
check(analyzeFieldValues({ shape: [2], data: Float32Array.from([5, 5]) }).constantFiniteField, "constant field");
check(analyzeFieldValues({ shape: [3], data: Float32Array.from([100_000, 0, -1]) }, "p").physicalViolationCount === 2, "positive physical-domain rule");
check(analyzeFieldValues({ shape: [3], data: Float32Array.from([0.01, 0, -0.001]) }, "qv").physicalViolationCount === 1, "nonnegative physical-domain rule");
const grid = { nr: 2, nth: 2, nz: 2, geometryKind: "physical" as const, coordinateSystem: "cartesian", scalarCoordinatesM: [[0, 1], [0, 1], [0, 1]] as [number[], number[], number[]] };
const steps: StepInfo[] = [
  { index: 0, simulationTimeS: 0, fields: ["w", "u"], grid, fieldUnits: { w: "m/s", u: "m/s" }, publicationState: "complete", publicationGeneration: "a" },
  { index: 1, simulationTimeS: 30, fields: ["w", "theta"], grid, fieldUnits: { w: "m/s", theta: "" }, publicationState: "complete", publicationGeneration: "b" },
];
const structure = analyzeDatasetStructure(steps);
check(structure.unionFieldCount === 3 && structure.sharedFieldCount === 1 && structure.varyingFieldCount === 2, "field coverage");
check(structure.exactCoordinateFrameCount === 2 && structure.time.expectedCadenceS === 30, "coordinate and time health");
const invalidCoordinates = analyzeDatasetStructure([{ ...steps[0], grid: { ...grid, scalarCoordinatesM: [[0, 1], [1, 0], [0, 1]] } }]);
check(invalidCoordinates.exactCoordinateFrameCount === 0, "exact coordinates require matching strictly increasing axes");
check(structure.fieldsMissingUnits.join() === "theta", "missing units");
check(structure.unitCompatibilityIssues.length === 0, "compatible field-unit families");
const incompatibleUnits = analyzeDatasetStructure([{ ...steps[0], fieldUnits: { w: "m/s", u: "knots" } }]);
check(incompatibleUnits.unitCompatibilityIssues[0]?.family === "wind components", "cross-field unit-family conflict");
const dataset = createLocalDatasetSource("cm1_netcdf", "/case/cm1out.nc");
const report = createDatasetFieldHealthReport({ dataset, fieldName: "w", unit: "m/s", requestedScope: "all_available_frames", requestedFrameCount: 2, missingFieldFrameCount: 0, structure, frames: [
  { ...analyzeFieldValues({ shape: [2], data: Float32Array.from([1, 3]) }), shape: [2], stepIndex: 0, simulationTimeS: 0 },
  { ...analyzeFieldValues({ shape: [2], data: Float32Array.from([5, 7]) }), shape: [2], stepIndex: 1, simulationTimeS: 30 },
] });
check(report.elementCount === 4 && report.mean === 4 && report.minimum === 1 && report.maximum === 7, "aggregate weighting");
check(report.schema === "nimbus.studio.data-health" && report.datasetIdentity.includes("cm1_netcdf"), "report identity");
console.log(`data health: ${passed} checks passed, ${failures.length} failed`);
if (failures.length) { failures.forEach((failure) => console.error(`FAIL ${failure}`)); process.exit(1); }
