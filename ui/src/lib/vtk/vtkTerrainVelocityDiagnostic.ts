import type { Npy3D } from "@/lib/platform/npy";
import {
  lambda2FromGradient,
  qCriterionFromGradient,
  swirlingStrengthFromGradient,
  type VelocityGradientDiagnostic,
} from "./vtkPreparation";

function solveSymmetric3(matrix: Float64Array, rhs: Float64Array): [number, number, number] | null {
  const [a, b, c, , d, e, , , f] = matrix;
  const determinant = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
  const scale = Math.max(Math.abs(a), Math.abs(d), Math.abs(f), 1);
  if (!Number.isFinite(determinant) || Math.abs(determinant) <= scale ** 3 * 1e-12) return null;
  return [
    ((d * f - e * e) * rhs[0] + (c * e - b * f) * rhs[1] + (b * e - c * d) * rhs[2]) / determinant,
    ((c * e - b * f) * rhs[0] + (a * f - c * c) * rhs[1] + (b * c - a * e) * rhs[2]) / determinant,
    ((b * e - c * d) * rhs[0] + (b * c - a * e) * rhs[1] + (a * d - b * b) * rhs[2]) / determinant,
  ];
}

export function terrainVelocityGradientDiagnostic(
  fields: readonly [Npy3D, Npy3D, Npy3D],
  diagnostic: VelocityGradientDiagnostic,
): Npy3D {
  const reference = fields[0];
  const geometry = reference.windowGeometry;
  const centers = geometry?.centroidsM;
  const volumes = geometry?.volumesM3;
  const samples = reference.data.length;
  if (!centers || !volumes || geometry?.weighting !== "physical_cell_volume"
      || fields.some((field) => field.shape.join("x") !== reference.shape.join("x")
        || field.data.length !== samples)
      || [centers.x, centers.y, centers.z, volumes].some((array) => array.length !== samples)) {
    throw new Error("Terrain velocity diagnostics require colocated physical centroids and volumes");
  }
  const [nx, ny, nz] = reference.shape;
  const output = new Float32Array(samples);
  const index = (x: number, y: number, z: number) => (x * ny + y) * nz + z;
  const directions = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]] as const;
  for (let x = 0; x < nx; x += 1) for (let y = 0; y < ny; y += 1) for (let z = 0; z < nz; z += 1) {
    const center = index(x, y, z);
    if (fields.some((field) => !Number.isFinite(field.data[center]))) {
      output[center] = Number.NaN;
      continue;
    }
    const matrix = new Float64Array(9);
    const rhs = [new Float64Array(3), new Float64Array(3), new Float64Array(3)];
    let neighbors = 0;
    for (const [di, dj, dk] of directions) {
      const xi = x + di; const yj = y + dj; const zk = z + dk;
      if (xi < 0 || xi >= nx || yj < 0 || yj >= ny || zk < 0 || zk >= nz) continue;
      const neighbor = index(xi, yj, zk);
      const delta = [centers.x[neighbor] - centers.x[center], centers.y[neighbor] - centers.y[center], centers.z[neighbor] - centers.z[center]];
      const lengthSquared = delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2;
      if (!(lengthSquared > 0) || !Number.isFinite(lengthSquared)
          || !Number.isFinite(volumes[neighbor]) || volumes[neighbor] <= 0
          || fields.some((field) => !Number.isFinite(field.data[neighbor]))) continue;
      const weight = 1 / lengthSquared;
      for (let row = 0; row < 3; row += 1) for (let column = 0; column < 3; column += 1) {
        matrix[row * 3 + column] += weight * delta[row] * delta[column];
      }
      for (let component = 0; component < 3; component += 1) {
        const difference = fields[component].data[neighbor] - fields[component].data[center];
        for (let axis = 0; axis < 3; axis += 1) rhs[component][axis] += weight * delta[axis] * difference;
      }
      neighbors += 1;
    }
    const gradient = new Float64Array(9);
    let valid = neighbors >= 3;
    for (let component = 0; component < 3 && valid; component += 1) {
      const solved = solveSymmetric3(matrix, rhs[component]);
      if (!solved) { valid = false; break; }
      gradient.set(solved, component * 3);
    }
    output[center] = !valid ? Number.NaN : diagnostic === "q-criterion"
      ? qCriterionFromGradient(gradient)
      : diagnostic === "lambda-2"
        ? lambda2FromGradient(gradient)
        : swirlingStrengthFromGradient(gradient);
  }
  return { shape: [...reference.shape], data: output, windowGeometry: geometry };
}
