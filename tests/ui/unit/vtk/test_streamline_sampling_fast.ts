import assert from "node:assert/strict";
import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData.js";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray.js";
import vtkImageStreamline from "@kitware/vtk.js/Filters/General/ImageStreamline.js";
import { finiteImageStreamline } from "../../../../ui/src/lib/vtk/finiteImageStreamline.ts";

const image = vtkImageData.newInstance();
const dims = [17, 13, 11] as const;
const origin = [-2300, 810, 120] as const;
const spacing = [125, 175, 90] as const;
image.setDimensions(...dims);
image.setOrigin([...origin]);
image.setSpacing([...spacing]);
const values = Float32Array.from({ length: dims[0] * dims[1] * dims[2] * 3 }, (_, i) => 4 * Math.sin(i * 0.03) + Math.cos(i * 0.14));
const vectors = vtkDataArray.newInstance({ name: "vectors", numberOfComponents: 3, values });
image.getPointData().setVectors(vectors);
const reference = vtkImageStreamline.newInstance();
const cached = finiteImageStreamline({}, image);
const positions: number[][] = [];
for (let i = 0; i < 2000; i++) positions.push(dims.map((n, axis) => origin[axis] + spacing[axis] * (n - 1) * ((i * (31 + 2 * axis)) % 1999) / 1998));
for (const x of [0, dims[0] - 1]) for (const y of [0, dims[1] - 1]) for (const z of [0, dims[2] - 1]) positions.push([origin[0] + x * spacing[0], origin[1] + y * spacing[1], origin[2] + z * spacing[2]]);
positions.push([origin[0] - 1, origin[1], origin[2]]);
for (const position of positions) {
  const expected = new Float32Array(3), actual = new Float32Array(3);
  const accepted = reference.vectorAt(position, vectors, image, expected);
  assert.equal(cached.vectorAt(position, vectors, image, actual), accepted, JSON.stringify({position, origin:image.getOrigin(),spacing:image.getSpacing(),extent:image.getExtent()}));
  if (accepted) assert.deepEqual(actual, expected);
}
for (const duration of [-0.3, 0.3]) for (const position of positions.slice(5, 45)) {
  const expected = Float32Array.from(position), actual = Float32Array.from(position);
  for (let step = 0; step < 80; step++) {
    const accepted = reference.computeNextStep(vectors, image, duration, expected as unknown as number[]);
    assert.equal(cached.computeNextStep(vectors, image, duration, actual as unknown as number[]), accepted);
    assert.deepEqual(actual, expected);
    if (!accepted) break;
  }
}
const timed = (sampler: typeof reference) => {
  const vector = new Float32Array(3);
  const start = performance.now();
  for (let repeat = 0; repeat < 10; repeat++) for (const position of positions) sampler.vectorAt(position, vectors, image, vector);
  return performance.now() - start;
};
timed(reference); timed(cached);
console.log(JSON.stringify({ samples: positions.length * 10, referenceMs: timed(reference), cachedMs: timed(cached) }));
cached.delete(); reference.delete(); vectors.delete(); image.delete();
console.log("Cached streamline sampling preserves float32 samples and forward/backward RK2 paths exactly.");
