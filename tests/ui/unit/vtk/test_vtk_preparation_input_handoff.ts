import { handoffVtkPreparationInput } from "../../../../ui/src/lib/vtk/vtkPreparationInputHandoff.ts";
import type { ScalarPreparationRequest, VectorPreparationRequest } from "../../../../ui/src/lib/vtk/vtkPreparation.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(`FAIL ${label}`);
}

const shape = [64, 64, 161] as [number, number, number];
const values = Float32Array.from(
  { length: shape[0] * shape[1] * shape[2] },
  (_, index) => index / 17,
);
const centroids = new Float64Array([1, 2, 3, 4]);
const scalar: ScalarPreparationRequest = {
  kind: "scalar",
  field: {
    shape,
    data: values,
    windowGeometry: {
      weighting: "physical_cell_volume",
      sampleLocation: "physical_volume_centroid",
      centroidsM: { x: centroids, y: centroids, z: centroids },
      volumesM3: centroids,
    },
  },
  grid: { drM: 1, dyM: 1, dzM: 1 },
};
const scalarHandoff = await handoffVtkPreparationInput(scalar);
check(
  scalar.field.data.byteLength > 2 * 1024 * 1024
    && scalar.field.data.buffer !== scalarHandoff.request.field.data.buffer,
  "chunked scalar handoff retains the decoded source buffer",
);
check(
  scalarHandoff.request.field.data[12345] === scalar.field.data[12345]
    && scalarHandoff.request.field.windowGeometry!.centroidsM!.x[3] === 4,
  "chunked scalar handoff preserves field and terrain geometry values",
);
check(
  scalarHandoff.transfer.length === 2
    && scalarHandoff.copiedBytes === values.byteLength + centroids.byteLength,
  "handoff deduplicates aliased terrain buffers in its transfer set",
);

const vector: VectorPreparationRequest = {
  kind: "vector",
  fields: [scalar.field, scalar.field, scalar.field],
  grid: scalar.grid,
};
const vectorHandoff = await handoffVtkPreparationInput(vector);
check(
  vectorHandoff.transfer.length === 2
    && vectorHandoff.request.fields.every((field) =>
      field.data.buffer === vectorHandoff.request.fields[0].data.buffer),
  "vector handoff preserves shared-buffer aliasing without duplicate copies",
);

let cancellationChecks = 0;
check(
  await handoffVtkPreparationInput(scalar, () => ++cancellationChecks > 1).then(
    () => false,
    (error) => error instanceof DOMException && error.name === "AbortError",
  ),
  "chunked handoff observes cancellation between bounded copy tasks",
);

console.log(`\nVTK preparation input handoff tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length) {
  console.log(failures.join("\n"));
  process.exit(1);
}
