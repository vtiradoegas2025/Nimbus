import {
  prepareVtkRequest,
  type VtkPreparationRequest,
} from "../../../../ui/src/lib/vtk/vtkPreparation.ts";
import {
  vtkPreparationInputBytes,
  vtkPreparationOutputUpperBoundBytes,
  vtkPreparationResourcePlan,
  vtkPreparationResultBytes,
} from "../../../../ui/src/lib/vtk/vtkPreparationResourcePlanning.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(`FAIL ${label}`);
}

const field = {
  shape: [4, 3, 2] as [number, number, number],
  data: Float32Array.from({ length: 24 }, (_, index) => index / 10),
};
const grid = {
  coordinateSystem: "cartesian" as const,
  drM: 100,
  dyM: 100,
  dzM: 100,
};

const requests: VtkPreparationRequest[] = [
  { kind: "scalar", field, grid, maxVoxels: 24 },
  { kind: "vector", fields: [field, field, field], grid, maxVoxels: 24 },
  {
    kind: "velocity-diagnostic",
    diagnostic: "q-criterion",
    fields: [field, field, field],
    grid,
    maxVoxels: 24,
  },
  {
    kind: "cloud",
    field,
    grid,
    volumeMaxVoxels: 24,
    surfaceMaxVoxels: 24,
    fidelityMode: "scientific",
  },
];

for (const request of requests) {
  const result = prepareVtkRequest(request as never);
  const predicted = vtkPreparationOutputUpperBoundBytes(request);
  const actual = vtkPreparationResultBytes(result);
  check(actual <= predicted, `${request.kind} prepared output stays under its admitted upper bound`);
  const plan = vtkPreparationResourcePlan(request, `test:${request.kind}`);
  check(
    plan.expectedPeakBytes.preparedMain === predicted
      && plan.expectedPeakBytes.workerResident === vtkPreparationInputBytes(request),
    `${request.kind} plan carries input and output ownership across stages`,
  );
}

const beastScalar: VtkPreparationRequest = {
  kind: "scalar",
  field: { shape: [256, 256, 160], data: new Float32Array(0) },
  grid,
  maxVoxels: 16_000_000,
};
const beastVector: VtkPreparationRequest = {
  kind: "vector",
  fields: [beastScalar.field, beastScalar.field, beastScalar.field],
  grid,
  maxVoxels: 16_000_000,
};
check(
  vtkPreparationOutputUpperBoundBytes(beastScalar) === 50 * 1024 * 1024,
  "Beast reduced scalar plan admits its 40 MiB lattice and optional 10 MiB support mask",
);
check(
  vtkPreparationOutputUpperBoundBytes(beastVector) === 120 * 1024 * 1024 + 6_144,
  "Beast reduced vector plan includes the observed 120 MiB lattice and bounded feature seeds",
);

const terrainGeometry = new Float64Array(24 * 4);
const terrainRequest: VtkPreparationRequest = {
  kind: "scalar",
  field: {
    ...field,
    windowGeometry: {
      weighting: "physical_cell_volume",
      sampleLocation: "physical_volume_centroid",
      centroidsM: {
        x: terrainGeometry.subarray(0, 24),
        y: terrainGeometry.subarray(24, 48),
        z: terrainGeometry.subarray(48, 72),
      },
      volumesM3: terrainGeometry.subarray(72, 96),
    },
  },
  grid: { ...grid, terrainFollowing: true },
  maxVoxels: 24,
};
check(
  vtkPreparationInputBytes(terrainRequest) === field.data.buffer.byteLength + terrainGeometry.byteLength,
  "terrain input admission counts each shared scalar and physical-geometry backing store once",
);
check(
  vtkPreparationOutputUpperBoundBytes(terrainRequest) === 24 * 17,
  "terrain resident output admits values, support, and three physical coordinates",
);

console.log(`\nVTK preparation resource planning tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length) {
  console.log("\nFailures:");
  for (const failure of failures) console.log(`  ${failure}`);
  process.exit(1);
}
