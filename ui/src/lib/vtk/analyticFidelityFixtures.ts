import type { Npy3D } from "@/lib/platform/npy";

export const ANALYTIC_FIDELITY_FIELD_NAMES = [
  "analytic_smooth_gradient",
  "analytic_thin_sheet",
  "analytic_helical_tube",
  "analytic_signed_couplet",
  "analytic_stratified_anomaly",
] as const;

export type AnalyticFidelityFieldName = typeof ANALYTIC_FIDELITY_FIELD_NAMES[number];

export interface AnalyticFidelityFixture {
  fieldName: AnalyticFidelityFieldName;
  field: Npy3D;
  spacingM: [number, number, number];
  expected: {
    min: number;
    max: number;
    nonzeroCount: number;
    signChanges: boolean;
    horizontallyUniform: boolean;
  };
}

function index(shape: readonly [number, number, number], x: number, y: number, z: number): number {
  return (x * shape[1] + y) * shape[2] + z;
}

function normalizedCoordinate(position: number, count: number): number {
  return count > 1 ? (2 * position) / (count - 1) - 1 : 0;
}

function summarize(
  fieldName: AnalyticFidelityFieldName,
  field: Npy3D,
  spacingM: [number, number, number],
): AnalyticFidelityFixture {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let nonzeroCount = 0;
  for (const value of field.data) {
    if (value < min) min = value;
    if (value > max) max = value;
    if (value !== 0) nonzeroCount += 1;
  }
  let horizontallyUniform = true;
  for (let z = 0; z < field.shape[2] && horizontallyUniform; z += 1) {
    const reference = field.data[index(field.shape, 0, 0, z)];
    for (let x = 0; x < field.shape[0] && horizontallyUniform; x += 1) {
      for (let y = 0; y < field.shape[1]; y += 1) {
        if (field.data[index(field.shape, x, y, z)] !== reference) {
          horizontallyUniform = false;
          break;
        }
      }
    }
  }
  return {
    fieldName,
    field,
    spacingM,
    expected: {
      min,
      max,
      nonzeroCount,
      signChanges: min < 0 && max > 0,
      horizontallyUniform,
    },
  };
}

export function createAnalyticFidelityFixture(
  fieldName: AnalyticFidelityFieldName,
  shape: [number, number, number] = [64, 64, 48],
  spacingM: [number, number, number] = [250, 250, 250],
): AnalyticFidelityFixture {
  if (shape.some((count) => !Number.isInteger(count) || count < 3)) {
    throw new Error("Analytic fidelity fixtures require at least three cells on every axis.");
  }
  const data = new Float32Array(shape[0] * shape[1] * shape[2]);
  const write = (x: number, y: number, z: number, value: number) => {
    data[index(shape, x, y, z)] = value;
  };

  if (fieldName === "analytic_smooth_gradient") {
    for (let x = 0; x < shape[0]; x += 1) {
      for (let y = 0; y < shape[1]; y += 1) {
        for (let z = 0; z < shape[2]; z += 1) write(x, y, z, z / (shape[2] - 1));
      }
    }
  } else if (fieldName === "analytic_thin_sheet") {
    const sheetZ = Math.floor(shape[2] * 0.55);
    for (let x = 0; x < shape[0]; x += 1) {
      for (let y = 0; y < shape[1]; y += 1) write(x, y, sheetZ, 1);
    }
  } else if (fieldName === "analytic_helical_tube") {
    for (let z = 0; z < shape[2]; z += 1) {
      const phase = (z / (shape[2] - 1)) * Math.PI * 3;
      const centerX = Math.cos(phase) * 0.48;
      const centerY = Math.sin(phase) * 0.48;
      for (let x = 0; x < shape[0]; x += 1) {
        const nx = normalizedCoordinate(x, shape[0]);
        for (let y = 0; y < shape[1]; y += 1) {
          const ny = normalizedCoordinate(y, shape[1]);
          const radius2 = ((nx - centerX) / 0.12) ** 2 + ((ny - centerY) / 0.12) ** 2;
          write(x, y, z, Math.exp(-0.5 * radius2));
        }
      }
    }
  } else if (fieldName === "analytic_signed_couplet") {
    for (let x = 0; x < shape[0]; x += 1) {
      const nx = normalizedCoordinate(x, shape[0]);
      for (let y = 0; y < shape[1]; y += 1) {
        const ny = normalizedCoordinate(y, shape[1]);
        const positive = Math.exp(-0.5 * (((nx + 0.32) / 0.18) ** 2 + (ny / 0.24) ** 2));
        const negative = Math.exp(-0.5 * (((nx - 0.32) / 0.18) ** 2 + (ny / 0.24) ** 2));
        for (let z = 0; z < shape[2]; z += 1) {
          const vertical = Math.sin(Math.PI * z / (shape[2] - 1));
          write(x, y, z, (positive - negative) * vertical);
        }
      }
    }
  } else {
    for (let x = 0; x < shape[0]; x += 1) {
      const nx = normalizedCoordinate(x, shape[0]);
      for (let y = 0; y < shape[1]; y += 1) {
        const ny = normalizedCoordinate(y, shape[1]);
        for (let z = 0; z < shape[2]; z += 1) {
          const nz = z / (shape[2] - 1);
          const anomaly = 0.24 * Math.exp(-0.5 * (
            (nx / 0.22) ** 2
            + (ny / 0.3) ** 2
            + ((nz - 0.45) / 0.12) ** 2
          ));
          write(x, y, z, 0.35 + 0.55 * nz + anomaly);
        }
      }
    }
  }

  return summarize(fieldName, { shape: [...shape], data }, [...spacingM]);
}

export function isAnalyticFidelityFieldName(value: string): value is AnalyticFidelityFieldName {
  return ANALYTIC_FIDELITY_FIELD_NAMES.some((fieldName) => fieldName === value);
}
