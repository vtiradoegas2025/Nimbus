import { terrainVelocityGradientDiagnostic } from "../../../../ui/src/lib/vtk/vtkTerrainVelocityDiagnostic.ts";
import type { Npy3D } from "../../../../ui/src/lib/platform/npy.ts";

const shape: [number, number, number] = [4, 4, 4];
const count = 64;
const x = new Float64Array(count); const y = new Float64Array(count);
const z = new Float64Array(count); const volumes = new Float64Array(count).fill(1);
const u = new Float32Array(count); const v = new Float32Array(count); const w = new Float32Array(count);
const omega = 0.02;
for (let i = 0; i < 4; i += 1) for (let j = 0; j < 4; j += 1) for (let k = 0; k < 4; k += 1) {
  const at = (i * 4 + j) * 4 + k;
  x[at] = i * 100; y[at] = j * 120; z[at] = k * 80 + i * 7 + j * 3;
  u[at] = -omega * y[at]; v[at] = omega * x[at]; w[at] = 0;
}
const windowGeometry: NonNullable<Npy3D["windowGeometry"]> = {
  aggregation: "cell_mean", weighting: "physical_cell_volume",
  sampleLocation: "physical_volume_centroid", centroidsM: { x, y, z }, volumesM3: volumes,
  sourceShape: shape, origin: [0, 0, 0], extent: shape, stride: [1, 1, 1], sourceFormat: "npy",
};
const field = (data: Float32Array): Npy3D => ({ shape, data, windowGeometry });
const fields = [field(u), field(v), field(w)] as [Npy3D, Npy3D, Npy3D];
for (const [diagnostic, expected, tolerance] of [
  ["q-criterion", omega ** 2, 1e-8],
  ["lambda-2", -(omega ** 2), 1e-8],
  ["swirling-strength", omega, 1e-6],
] as const) {
  const result = terrainVelocityGradientDiagnostic(fields, diagnostic);
  const error = Math.max(...result.data.map((value) => Math.abs(value - expected)));
  if (error > tolerance) throw new Error(`${diagnostic} physical-gradient error ${error}`);
  if (result.windowGeometry !== windowGeometry) throw new Error(`${diagnostic} lost terrain geometry`);
}
const singular = field(u);
singular.windowGeometry = { ...windowGeometry, centroidsM: { x: new Float64Array(count), y: new Float64Array(count), z: new Float64Array(count) } };
const invalid = terrainVelocityGradientDiagnostic([singular, { ...field(v), windowGeometry: singular.windowGeometry }, { ...field(w), windowGeometry: singular.windowGeometry }], "q-criterion");
if (!invalid.data.every(Number.isNaN)) throw new Error("singular terrain metrics did not fail closed");
console.log("\nterrain velocity diagnostic tests: 7 passed, 0 failed.");
