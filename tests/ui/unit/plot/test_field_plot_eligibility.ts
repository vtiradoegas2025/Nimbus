import {
  addPlotFieldCapabilities,
  fieldsAcrossStepsForPlotRanks,
  fieldsForPlotRanks,
  fieldsSharePlotRank,
  fieldsWithSharedPlotRank,
} from "../../../../ui/src/lib/plot/fieldPlotEligibility.ts";
import {
  CROSS_SECTION_VECTOR_CAPABILITY,
  NATIVE_REDUCTION_CAPABILITY,
  NATIVE_VOLUME_REDUCTION_CAPABILITY,
  PLANE_SCALAR_CAPABILITY,
  PLAN_VECTOR_CAPABILITY,
  VOLUME_SCALAR_CAPABILITY,
} from "../../../../ui/src/lib/plot/plotRegistry.ts";
import {
  isVolumetricField,
  NIMBUS_OUTPUT_FIELDS,
} from "../../../../ui/src/lib/plot/fieldProvenance.ts";
import {
  clearFieldCache,
  registerNetcdfFrame,
  type StepInfo,
} from "../../../../ui/src/lib/run/runOutputs.ts";

let passed = 0;
let failed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else {
    failed += 1;
    failures.push(`FAIL ${label}`);
  }
}

const runDir = "field-plot-contract://synthetic";
const generation = "netcdf:field-plot-contract";
const nativeOrigins = {
  w: "native",
  u: "native",
  v: "native",
  cape: "native",
  profile: "native",
  expression3d: "user_expression",
} as const;

function step(index: number): StepInfo {
  return {
    index,
    simulationTimeS: index * 10,
    fields: ["u", "v", "w", "cape", "profile", "expression3d"],
    grid: { nr: 4, nth: 3, nz: 2 },
    fieldUnits: { w: "m/s", cape: "J/kg" },
    fieldOrigins: nativeOrigins,
    publicationState: "complete",
    publicationGeneration: generation,
  };
}

const steps = [step(0), step(1)];
for (const current of steps) {
  registerNetcdfFrame(
    runDir,
    current.index,
    generation,
    `/synthetic/frame-${current.index}.nc`,
    current.index,
    [4, 3, 2],
    {
      w: [4, 3, 2],
      u: [4, 3, 2],
      v: [4, 3, 2],
      cape: [4, 3],
      profile: [2],
      expression3d: [4, 3, 2],
    },
    ["u", "v", "w", "cape", "profile"],
  );
}

check(
  fieldsForPlotRanks(runDir, steps[0], [2, 3]).join(",") === "u,v,w,cape,expression3d",
  "plane eligibility accepts rank 2/3 and rejects rank 1",
);
check(
  fieldsForPlotRanks(runDir, steps[0], [3]).join(",") === "u,v,w,expression3d",
  "volume eligibility accepts only exact rank 3",
);
check(
  fieldsForPlotRanks(runDir, steps[0], [1]).join(",") === "profile",
  "rank-1 eligibility is explicit rather than inferred from a name",
);
check(
  fieldsSharePlotRank(runDir, steps[0], ["w", "expression3d"], [3]),
  "matched rank-3 components qualify",
);
check(
  !fieldsSharePlotRank(runDir, steps[0], ["w", "cape"], [2, 3]),
  "mixed-rank components fail closed",
);
check(
  fieldsWithSharedPlotRank(
    ["u", "cape", "profile"],
    [{ runDir, step: steps[0] }, { runDir, step: steps[1] }],
    [2, 3],
  ).join(",") === "u,cape",
  "comparison and ensemble fields require one shared spatial rank",
);

const nativeOnly = new Set(["native"] as const);
check(
  fieldsAcrossStepsForPlotRanks(runDir, steps, [2, 3], nativeOnly).join(",")
    === "cape,u,v,w",
  "native reducers exclude provider-backed and non-grid fields",
);
check(
  fieldsAcrossStepsForPlotRanks(runDir, steps, [3], nativeOnly).join(",") === "u,v,w",
  "native axis reducers retain only rank-3 history",
);

const capabilities = addPlotFieldCapabilities(new Set(steps[0].fields), runDir, steps);
for (const capability of [
  PLANE_SCALAR_CAPABILITY,
  VOLUME_SCALAR_CAPABILITY,
  NATIVE_REDUCTION_CAPABILITY,
  NATIVE_VOLUME_REDUCTION_CAPABILITY,
  PLAN_VECTOR_CAPABILITY,
  CROSS_SECTION_VECTOR_CAPABILITY,
]) {
  check(capabilities.has(capability), `capability set includes ${capability}`);
}
check(
  !steps[0].fields.some((field) => field.startsWith("__")),
  "capability sentinels never enter the field catalog",
);

// Exhaustive Nimbus catalog matrix: every scalar field is a valid plan map;
// only fields whose output contract carries z may enter profiles, vertical
// cross-sections, volume rendering, or marching cubes.
const catalogRunDir = "field-plot-contract://nimbus-catalog";
const catalogGeneration = "netcdf:nimbus-field-catalog";
const catalogFields = Object.keys(NIMBUS_OUTPUT_FIELDS);
const catalogShapes = Object.fromEntries(catalogFields.map((field) => [
  field,
  isVolumetricField(field) ? [4, 3, 2] : [4, 3],
]));
const catalogStep: StepInfo = {
  index: 0,
  simulationTimeS: 0,
  fields: catalogFields,
  grid: { nr: 4, nth: 3, nz: 2 },
  fieldUnits: {},
  fieldOrigins: Object.fromEntries(catalogFields.map((field) => [field, "native"])),
  publicationState: "complete",
  publicationGeneration: catalogGeneration,
};
registerNetcdfFrame(
  catalogRunDir,
  0,
  catalogGeneration,
  "/synthetic/nimbus-catalog.nc",
  0,
  [4, 3, 2],
  catalogShapes,
  catalogFields,
);
const catalogPlane = new Set(fieldsForPlotRanks(catalogRunDir, catalogStep, [2, 3]));
const catalogVolume = new Set(fieldsForPlotRanks(catalogRunDir, catalogStep, [3]));
check(catalogPlane.size === catalogFields.length, "every Nimbus scalar contract is eligible for a plan map");
for (const field of catalogFields) {
  check(
    catalogVolume.has(field) === isVolumetricField(field),
    `${field} enters 3D exactly when its catalog contract carries z`,
  );
}

clearFieldCache();
for (const failure of failures) console.error(failure);
console.log(`\nfield plot eligibility tests: ${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
