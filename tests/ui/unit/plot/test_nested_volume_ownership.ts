import { planNestedVolumeOwnership, type OutputHierarchySelection } from "../../../../ui/src/lib/run/outputHierarchy.ts";
import { prepareVtkRequest } from "../../../../ui/src/lib/vtk/vtkPreparation.ts";

const rootStep = {
  index: 0, simulationTimeS: 0, fields: ["temperature"], publicationState: "complete" as const,
  publicationGeneration: "g",
  fieldUnits: { temperature: "K" },
  grid: { coordinateSystem: "cartesian", drM: 1, dyM: 1, dzM: 1,
    physicalBoundsM: { lower: [0, 0, 0] as [number, number, number], upper: [4, 4, 4] as [number, number, number] } },
};
const childStep = {
  ...rootStep,
  grid: { ...rootStep.grid, drM: 0.5, dyM: 0.5,
    physicalBoundsM: { lower: [1, 1, 1] as [number, number, number], upper: [3, 3, 3] as [number, number, number] } },
};
const root = { levelId: "root" as const, parentLevelId: null, runDir: "/root", step: rootStep, depth: 0, horizontalCellAreaM2: 1 };
const child = { levelId: "child_0000", parentLevelId: "root" as const, runDir: "/child", step: childStep, depth: 1, horizontalCellAreaM2: 0.25 };
const selection: OutputHierarchySelection = {
  level: child,
  chain: [root, child],
  requestedLevelId: "auto",
  disclosure: "child",
  fallbackReason: null,
};
const plan = planNestedVolumeOwnership(selection, "temperature");
if (plan.length !== 2 || plan[0].excludeChildBoundsM?.lower.join(",") !== "1,1,1"
    || plan[1].excludeChildBoundsM !== null) {
  throw new Error("Nested volume ownership did not assign parent exclusion and child ownership");
}

const field = { shape: [4, 4, 4] as [number, number, number], data: new Float32Array(64).fill(5) };
const prepared = prepareVtkRequest({
  kind: "scalar",
  field,
  grid: { coordinateSystem: "cartesian", drM: 1, dyM: 1, dzM: 1, scalarOriginM: [0.5, 0.5, 0.5] },
  excludePhysicalBoundsM: { lower: [1, 1, 1], upper: [3, 3, 3] },
});
let excluded = 0;
let retained = 0;
for (const value of prepared.values) {
  if (value === 0) excluded += 1;
  if (value === 5) retained += 1;
}
if (excluded !== 8 || retained !== 56 || field.data.some((value) => value !== 5)) {
  throw new Error("Parent scalar exclusion is not exact or mutated its scientific source");
}

let rejected = false;
try {
  planNestedVolumeOwnership({ ...selection, chain: [root, { ...child, step: { ...childStep, fields: [] } }] }, "temperature");
} catch { rejected = true; }
if (!rejected) throw new Error("Nested ownership accepted a level missing the selected field");

console.log("\nnested volume ownership tests: 3 passed, 0 failed.");
