import { createLocalDatasetSource } from "../../../../ui/src/lib/dataset/datasetSources.ts";
import {
  DATASET_ALIGNMENT_SCHEMA,
  comparisonUnitKey,
  alignDatasetGrids,
  alignDatasetTimes,
  alignDatasets,
} from "../../../../ui/src/lib/dataset/datasetAlignment.ts";
import type { GridInfo, StepInfo } from "../../../../ui/src/lib/run/runOutputs.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(label);
}

const grid = (x = [0, 100, 250], staggering = "cell_centered"): GridInfo => ({
  nr: 3,
  nth: 2,
  nz: 2,
  coordinateSystem: "cartesian",
  geometryKind: "physical",
  staggering,
  scalarCoordinatesM: [x, [0, 200], [0, 75]],
  scalarOriginM: [x[0], 0, 0],
});

const frames = (
  times: readonly (number | null)[],
  fields: readonly string[] = ["w", "theta"],
  units: Record<string, string> = { w: "m/s", theta: "K" },
  frameGrid: GridInfo = grid(),
): StepInfo[] => times.map((time, index) => ({
  index,
  simulationTimeS: time,
  fields: [...fields],
  grid: frameGrid,
  fieldUnits: { ...units },
  publicationState: "complete",
  publicationGeneration: `fixture-${index}`,
}));

const cm1 = createLocalDatasetSource("cm1_netcdf", "/case/cm1out.nc");
const nimbus = createLocalDatasetSource("nimbus_run", "/case/nimbus");

check(comparisonUnitKey("m s-1") === comparisonUnitKey("m/s"), "algebraically identical velocity unit spellings align");
check(comparisonUnitKey("s^-1") === comparisonUnitKey("1/s"), "algebraically identical inverse-time spellings align");

{
  const report = alignDatasets(cm1, frames([0, 30, 60]), nimbus, frames([0, 30, 60]));
  check(report.schema === DATASET_ALIGNMENT_SCHEMA, "report schema is versioned");
  check(report.verdict === "exact" && report.differenceAllowed, "identical canonical datasets allow exact difference");
  check(report.time.matches.length === 3 && report.time.status === "exact", "all exact physical times pair");
  check(report.grid.status === "exact" && report.grid.shape?.join("x") === "3x2x2", "stretched scalar coordinates align exactly");
  check(report.differenceEligibleFields.join(",") === "theta,w", "matching declared-unit fields are difference eligible");
}

{
  const report = alignDatasets(cm1, frames([0]), cm1, frames([0]));
  check(report.verdict === "blocked" && report.blockers.some((blocker) => blocker.includes("same dataset")), "self-comparison is rejected by source identity");
}

{
  const moving = frames([0]);
  moving[0] = {
    ...moving[0],
    motion: { mode: "constant_velocity", earthOriginM: [0, 0], windsEarthRelative: true },
  };
  const report = alignDatasets(cm1, moving, nimbus, frames([0]));
  check(
    !report.differenceAllowed && report.blockers.some((blocker) => blocker.includes("translates across the Earth")),
    "cellwise comparison rejects a translating domain",
  );
}

{
  const left = frames([0, 30]);
  left[1] = {
    ...left[1],
    motion: { mode: "tracked_window", earthOriginM: [0, 0], windsEarthRelative: true },
  };
  const report = alignDatasets(cm1, left, nimbus, frames([0, 30]));
  check(
    !report.differenceAllowed && report.blockers.some((blocker) => blocker.includes("window follows the storm")),
    "matched-time comparison validates motion on every paired frame",
  );
}

{
  const left = frames([0]);
  const right = frames([0]);
  left[0] = { ...left[0], motion: { mode: "stationary", earthOriginM: [0, 0], windsEarthRelative: true } };
  right[0] = { ...right[0], motion: { mode: "stationary", earthOriginM: [1000, 0], windsEarthRelative: true } };
  const report = alignDatasets(cm1, left, nimbus, right);
  check(
    !report.differenceAllowed && report.blockers.some((blocker) => blocker.includes("different Earth origins")),
    "cellwise comparison rejects different Earth origins",
  );
}

{
  const report = alignDatasets(cm1, frames([0, 30, 60]), nimbus, frames([0, 60, 90]));
  check(report.verdict === "partial" && report.differenceAllowed, "partial time overlap allows only matched-frame difference");
  check(report.time.matches.length === 2 && report.time.unmatchedLeftFrameIndices.length === 1, "partial report inventories unmatched A frame");
  check(report.time.unmatchedRightFrameIndices.length === 1, "partial report inventories unmatched B frame");
}

{
  const report = alignDatasets(cm1, frames([0, 30]), nimbus, frames([0, 0]));
  check(report.verdict === "blocked" && !report.differenceAllowed, "duplicate B times block ambiguous pairing");
  check(report.time.ambiguousTimeCount === 1, "duplicate timestamp count is explicit");
}

{
  const report = alignDatasets(cm1, frames([null]), nimbus, frames([0]));
  check(report.time.status === "blocked", "missing physical time blocks pairing");
  check(report.blockers.some((blocker) => blocker.includes("physical-time")), "missing-time blocker is actionable");
}

{
  const right = frames([0], ["w", "theta"], { w: "km/h", theta: "K" });
  const report = alignDatasets(cm1, frames([0]), nimbus, right);
  check(report.differenceEligibleFields.join(",") === "theta", "unequal unit syntax fails closed without conversion");
  check(report.unitMismatches[0]?.field === "w", "unit mismatch names the affected field");
}

{
  const report = alignDatasets(cm1, frames([0], ["w"]), nimbus, frames([0], ["theta"], { theta: "K" }));
  check(report.verdict === "blocked", "no common canonical fields blocks difference");
  check(report.leftOnlyFields[0] === "w" && report.rightOnlyFields[0] === "theta", "source-only fields are inventoried");
}

{
  const report = alignDatasetGrids(frames([0]), frames([0], ["w"], { w: "m/s" }, grid([0, 110, 250])));
  check(report.status === "blocked" && report.maxCoordinateDeltaM === 10, "coordinate delta blocks unqualified interpolation");
}

{
  const indexed = { ...grid(), geometryKind: "index" as const };
  check(alignDatasetGrids(frames([0]), frames([0], ["w"], { w: "m/s" }, indexed)).status === "blocked", "index geometry blocks physical difference");
}

{
  const legacyPhysical = { ...grid(), geometryKind: undefined, scalarCoordinatesM: undefined, drM: 125, dyM: 200, dzM: 75 };
  const uniformLeft = frames([0], ["w"], { w: "m/s" }, grid([0, 100, 200]));
  check(alignDatasetGrids(uniformLeft, frames([0], ["w"], { w: "m/s" }, legacyPhysical)).status === "blocked", "declared physical spacing still must match exact source coordinates");
  legacyPhysical.drM = 100;
  check(alignDatasetGrids(uniformLeft, frames([0], ["w"], { w: "m/s" }, legacyPhysical)).status === "exact", "legacy Cartesian spacing and origin imply physical geometry");
}

{
  const changing = frames([0, 30]);
  changing[1] = { ...changing[1], grid: grid([0, 101, 250]) };
  check(alignDatasetGrids(changing, frames([0, 30])).status === "blocked", "within-dataset grid changes block comparison");
}

{
  const report = alignDatasets(cm1, frames([0], ["w"]), nimbus, frames([0], ["w"], { w: "m/s" }, grid(undefined, "c_grid")));
  check(report.differenceAllowed && report.verdict === "partial", "staggering label difference is reviewable on exact scalar centers");
  check(report.warnings.some((warning) => warning.includes("staggering")), "staggering difference is disclosed");
}

{
  const time = alignDatasetTimes(frames([10]), frames([10 + 5e-7]));
  check(time.status === "exact" && time.matches.length === 1, "sub-microsecond serialization noise is tolerated");
}

console.log(`dataset alignment: ${passed} checks passed, ${failures.length} failed`);
if (failures.length > 0) {
  for (const failure of failures) console.error(`FAIL ${failure}`);
  process.exit(1);
}
