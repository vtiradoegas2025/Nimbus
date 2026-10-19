import type { CartesianVoxelGrid } from "@/lib/science/cartesianInterp";

export interface CloudDetailSummary {
  positiveCount: number;
  mass: number;
  mean: number;
  coefficientOfVariation: number;
  interiorGradientRms: number;
}

export interface PairedCloudDetailValidation {
  reference: CloudDetailSummary;
  reconstructed: CloudDetailSummary;
  presentation: CloudDetailSummary;
  supportLeakCount: number;
  relativeMassBias: number;
  variabilityGapBefore: number;
  variabilityGapAfter: number;
  gradientGapBefore: number;
  gradientGapAfter: number;
}

/** Distribution/spectrum validation for reference vs coarse reconstruction on same lattice. */
export function validatePairedCloudDetail(
  reference: CartesianVoxelGrid,
  reconstructed: CartesianVoxelGrid,
  presentation: CartesianVoxelGrid,
): PairedCloudDetailValidation {
  assertSameLattice(reference, reconstructed);
  assertSameLattice(reference, presentation);
  const referenceSummary = summarizeCloudDetail(reference);
  const reconstructedSummary = summarizeCloudDetail(reconstructed);
  const presentationSummary = summarizeCloudDetail(presentation);
  let supportLeakCount = 0;
  for (let i = 0; i < presentation.data.length; i++) {
    if (presentation.data[i] > 0 && !(reconstructed.data[i] > 0)) supportLeakCount++;
  }
  const massScale = Math.max(Math.abs(reconstructedSummary.mass), 1e-12);
  return {
    reference: referenceSummary,
    reconstructed: reconstructedSummary,
    presentation: presentationSummary,
    supportLeakCount,
    relativeMassBias: (presentationSummary.mass - reconstructedSummary.mass) / massScale,
    variabilityGapBefore: Math.abs(
      reconstructedSummary.coefficientOfVariation - referenceSummary.coefficientOfVariation,
    ),
    variabilityGapAfter: Math.abs(
      presentationSummary.coefficientOfVariation - referenceSummary.coefficientOfVariation,
    ),
    gradientGapBefore: Math.abs(
      reconstructedSummary.interiorGradientRms - referenceSummary.interiorGradientRms,
    ),
    gradientGapAfter: Math.abs(
      presentationSummary.interiorGradientRms - referenceSummary.interiorGradientRms,
    ),
  };
}

export function summarizeCloudDetail(grid: CartesianVoxelGrid): CloudDetailSummary {
  const { data } = grid;
  let positiveCount = 0;
  let mass = 0;
  let squareSum = 0;
  for (const value of data) {
    if (!(value > 0) || !Number.isFinite(value)) continue;
    positiveCount++;
    mass += value;
    squareSum += value * value;
  }
  const mean = positiveCount > 0 ? mass / positiveCount : 0;
  const variance = positiveCount > 0
    ? Math.max(0, squareSum / positiveCount - mean * mean)
    : 0;
  return {
    positiveCount,
    mass,
    mean,
    coefficientOfVariation: mean > 0 ? Math.sqrt(variance) / mean : 0,
    interiorGradientRms: interiorGradientRms(grid),
  };
}

function interiorGradientRms(grid: CartesianVoxelGrid): number {
  const [nx, ny, nz] = grid.shape;
  const [dx, dy, dz] = grid.spacing;
  let squareSum = 0;
  let count = 0;
  const accumulate = (a: number, b: number, spacing: number) => {
    if (!(a > 0) || !(b > 0) || !(spacing > 0)) return;
    const gradient = (b - a) / spacing;
    squareSum += gradient * gradient;
    count++;
  };
  for (let x = 0; x < nx; x++) {
    for (let y = 0; y < ny; y++) {
      for (let z = 0; z < nz; z++) {
        const index = (x * ny + y) * nz + z;
        if (x + 1 < nx) accumulate(grid.data[index], grid.data[index + ny * nz], dx);
        if (y + 1 < ny) accumulate(grid.data[index], grid.data[index + nz], dy);
        if (z + 1 < nz) accumulate(grid.data[index], grid.data[index + 1], dz);
      }
    }
  }
  return count > 0 ? Math.sqrt(squareSum / count) : 0;
}

function assertSameLattice(a: CartesianVoxelGrid, b: CartesianVoxelGrid): void {
  const same = a.shape.every((value, axis) => value === b.shape[axis])
    && a.spacing.every((value, axis) => value === b.spacing[axis])
    && a.origin.every((value, axis) => value === b.origin[axis])
    && a.data.length === b.data.length;
  if (!same) throw new Error("paired cloud-detail validation requires one registered display lattice");
}
