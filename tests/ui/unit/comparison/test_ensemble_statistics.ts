import { alignEnsembleMembers } from "../../../../ui/src/lib/comparison/ensembleAlignment.ts";
import { computeEnsembleCellStatistics, withEnsembleThreshold } from "../../../../ui/src/lib/comparison/ensembleStatistics.ts";
import { createLocalDatasetSource } from "../../../../ui/src/lib/dataset/datasetSources.ts";
import type { GridInfo, StepInfo } from "../../../../ui/src/lib/run/runOutputs.ts";

let passed = 0;
const failures: string[] = [];
const check = (condition: boolean, label: string) => condition ? passed += 1 : failures.push(label);
const close = (left: number, right: number) => Math.abs(left - right) < 1e-4;
const samples = [Float32Array.of(1, 2, Number.NaN), Float32Array.of(3, 4, 9), Float32Array.of(5, 6, 12)];
const moments = computeEnsembleCellStatistics(samples, 3);
const originalProbability = Array.from(moments.probabilityPercent);
for (const threshold of [-10, 0, 3, 4, 10]) {
  const updated = withEnsembleThreshold(samples, moments, threshold);
  const fresh = computeEnsembleCellStatistics(samples, threshold);
  check(updated.mean === moments.mean && updated.spread === moments.spread,
    `threshold ${threshold} retains exact completed means and spreads without another source read`);
  check(updated.probabilityPercent.every((value, index) => Object.is(value, fresh.probabilityPercent[index]))
    && close(updated.domainExceedancePercent, fresh.domainExceedancePercent),
  `threshold ${threshold} probabilities equal a complete fresh scientific calculation`);
  check(Number.isNaN(updated.probabilityPercent[2]) && updated.completeCellCount === 2,
    `threshold ${threshold} preserves strict common finite support`);
}
check(moments.probabilityPercent.every((value, index) => Object.is(value, originalProbability[index])),
  "new threshold calculations do not mutate the previous completed probability field");
for (const invalid of [Number.NaN, Infinity, -Infinity]) {
  let rejected = false;
  try { withEnsembleThreshold(samples, moments, invalid); } catch { rejected = true; }
  check(rejected, "a nonfinite threshold cannot fabricate a probability field");
}
let changedSourceRejected = false;
try { withEnsembleThreshold(samples.map(value => value.slice()), moments, 4); }
catch { changedSourceRejected = true; }
check(changedSourceRejected, "another source array set cannot reuse the old ensemble's mean and spread");

const stats = computeEnsembleCellStatistics([
  new Float32Array([1, 2, Number.NaN]),
  new Float32Array([3, 4, 9]),
  new Float32Array([5, 6, 12]),
], 3);
check(stats.completeCellCount === 2, "strict finite membership");
check(close(stats.mean[0], 3) && close(stats.mean[1], 4), "cell means");
check(close(stats.spread[0], 2) && close(stats.spread[1], 2), "sample spread uses N-1");
check(close(stats.probabilityPercent[0], 100 / 3) && close(stats.probabilityPercent[1], 200 / 3), "exceedance probability");
check(Number.isNaN(stats.mean[2]), "incomplete cells remain non-finite");

const grid: GridInfo = {
  nr: 2, nth: 2, nz: 2, coordinateSystem: "cartesian", geometryKind: "physical",
  scalarCoordinatesM: [[0, 1], [0, 1], [0, 1]], scalarOriginM: [0, 0, 0],
};
const frame = (time: number): StepInfo => ({
  index: time, simulationTimeS: time, fields: ["w", "reflectivity_dbz"],
  fieldUnits: { w: "m/s", reflectivity_dbz: "dBZ" }, grid,
  publicationState: "complete", publicationGeneration: `fixture-${time}`,
});
const sources = ["reference", "member-1", "member-2"].map((name) => createLocalDatasetSource("nimbus_run", `/case/${name}`));
const report = alignEnsembleMembers([
  { source: sources[0], steps: [frame(0), frame(30), frame(60)] },
  { source: sources[1], steps: [frame(0), frame(30), frame(60)] },
  { source: sources[2], steps: [frame(0), frame(60)] },
]);
check(report.aggregateAllowed && report.status === "partial", "qualified partial ensemble");
check(report.timeMatches.length === 2 && report.timeMatches[1].simulationTimeS === 60, "all-member time intersection");
check(report.eligibleFields.join(",") === "reflectivity_dbz,w", "all-member field intersection");
const duplicate = alignEnsembleMembers([
  { source: sources[0], steps: [frame(0)] },
  { source: sources[0], steps: [frame(0)] },
]);
check(!duplicate.aggregateAllowed && duplicate.blockers.some((item) => item.includes("distinct")), "duplicate sources fail closed");
const oversized = alignEnsembleMembers(Array.from({ length: 13 }, (_, index) => ({
  source: createLocalDatasetSource("nimbus_run", `/case/member-${index}`), steps: [frame(0)],
})));
check(!oversized.aggregateAllowed && oversized.blockers.some((item) => item.includes("bounded to 12")), "interactive member count is bounded");

console.log(`ensemble statistics: ${passed} checks passed, ${failures.length} failed`);
if (failures.length) { failures.forEach((failure) => console.error(`FAIL ${failure}`)); process.exit(1); }
