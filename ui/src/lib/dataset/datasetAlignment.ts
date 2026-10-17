import type { LocalDatasetSource } from "./datasetSources";
import { datasetSourceIdentity } from "./datasetSources";
import type { GridInfo, StepInfo } from "@/lib/run/runOutputs";
import { canCompareByCell } from "@/lib/run/frameMotion";

export const DATASET_ALIGNMENT_SCHEMA = "nimbus.studio.dataset-alignment" as const;
export const DATASET_ALIGNMENT_VERSION = 1 as const;

export type AlignmentStatus = "exact" | "partial" | "blocked";

export interface DatasetTimeMatch {
  leftFrameIndex: number;
  rightFrameIndex: number;
  simulationTimeS: number;
  deltaS: number;
}

export interface DatasetTimeAlignment {
  status: AlignmentStatus;
  toleranceS: number;
  matches: DatasetTimeMatch[];
  unmatchedLeftFrameIndices: number[];
  unmatchedRightFrameIndices: number[];
  ambiguousTimeCount: number;
  detail: string;
}

export interface DatasetGridAlignment {
  status: "exact" | "blocked";
  shape: [number, number, number] | null;
  maxCoordinateDeltaM: number | null;
  staggeringDiffers: boolean;
  detail: string;
}

export interface DatasetUnitMismatch {
  field: string;
  leftUnit: string | null;
  rightUnit: string | null;
  reason: string;
}

export interface DatasetAlignmentReport {
  schema: typeof DATASET_ALIGNMENT_SCHEMA;
  schemaVersion: typeof DATASET_ALIGNMENT_VERSION;
  leftDatasetIdentity: string;
  rightDatasetIdentity: string;
  verdict: AlignmentStatus;
  differenceAllowed: boolean;
  commonFields: string[];
  differenceEligibleFields: string[];
  leftOnlyFields: string[];
  rightOnlyFields: string[];
  unitMismatches: DatasetUnitMismatch[];
  grid: DatasetGridAlignment;
  time: DatasetTimeAlignment;
  blockers: string[];
  warnings: string[];
}

function finiteTime(frame: StepInfo): number | null {
  return frame.simulationTimeS != null && Number.isFinite(frame.simulationTimeS)
    ? frame.simulationTimeS
    : null;
}

function duplicateTimeCount(times: readonly number[], tolerance: number): number {
  const sorted = [...times].sort((left, right) => left - right);
  return sorted.slice(1).filter((time, index) => Math.abs(time - sorted[index]) <= tolerance).length;
}

export function alignDatasetTimes(
  leftFrames: readonly StepInfo[],
  rightFrames: readonly StepInfo[],
): DatasetTimeAlignment {
  const leftTimes = leftFrames.map(finiteTime);
  const rightTimes = rightFrames.map(finiteTime);
  const finite = [...leftTimes, ...rightTimes].filter((time): time is number => time != null);
  const scale = Math.max(1, ...finite.map(Math.abs));
  const toleranceS = Math.max(1e-6, scale * 1e-9);
  const ambiguousTimeCount = duplicateTimeCount(leftTimes.filter((time): time is number => time != null), toleranceS)
    + duplicateTimeCount(rightTimes.filter((time): time is number => time != null), toleranceS);
  const availableRight = new Set(rightFrames.map((_, index) => index));
  const matches: DatasetTimeMatch[] = [];
  const unmatchedLeftFrameIndices: number[] = [];
  for (let leftIndex = 0; leftIndex < leftTimes.length; leftIndex += 1) {
    const leftTime = leftTimes[leftIndex];
    if (leftTime == null) {
      unmatchedLeftFrameIndices.push(leftIndex);
      continue;
    }
    let bestRight = -1;
    let bestDelta = Number.POSITIVE_INFINITY;
    for (const rightIndex of availableRight) {
      const rightTime = rightTimes[rightIndex];
      if (rightTime == null) continue;
      const delta = Math.abs(leftTime - rightTime);
      if (delta < bestDelta) {
        bestDelta = delta;
        bestRight = rightIndex;
      }
    }
    if (bestRight >= 0 && bestDelta <= toleranceS) {
      availableRight.delete(bestRight);
      matches.push({
        leftFrameIndex: leftIndex,
        rightFrameIndex: bestRight,
        simulationTimeS: leftTime,
        deltaS: rightTimes[bestRight]! - leftTime,
      });
    } else {
      unmatchedLeftFrameIndices.push(leftIndex);
    }
  }
  const unmatchedRightFrameIndices = [...availableRight].sort((left, right) => left - right);
  const complete = matches.length === leftFrames.length
    && matches.length === rightFrames.length
    && ambiguousTimeCount === 0;
  const status: AlignmentStatus = complete
    ? "exact"
    : matches.length > 0 && ambiguousTimeCount === 0
      ? "partial"
      : "blocked";
  const detail = status === "exact"
    ? `${matches.length} physical times match one-to-one.`
    : status === "partial"
      ? `${matches.length} physical times match; ${unmatchedLeftFrameIndices.length} A and ${unmatchedRightFrameIndices.length} B frames are unmatched.`
      : ambiguousTimeCount > 0
        ? `${ambiguousTimeCount} duplicate timestamps make frame pairing ambiguous.`
        : "No unambiguous physical-time pairs are available.";
  return {
    status,
    toleranceS,
    matches,
    unmatchedLeftFrameIndices,
    unmatchedRightFrameIndices,
    ambiguousTimeCount,
    detail,
  };
}

function close(left: number, right: number): boolean {
  return Math.abs(left - right) <= Math.max(1e-6, Math.abs(left) * 1e-9, Math.abs(right) * 1e-9);
}

function validRange(grid: GridInfo, axis: number, count: number): [number, number] | null {
  const lower = grid.validIndexRange?.lower[axis] ?? 0;
  const upper = grid.validIndexRange?.upper[axis] ?? count - 1;
  return Number.isInteger(lower) && Number.isInteger(upper) && lower >= 0 && upper >= lower && upper < count
    ? [lower, upper]
    : null;
}

function axisCoordinates(grid: GridInfo, axis: number): number[] | null {
  const counts = [grid.nr, grid.nth, grid.nz];
  const count = counts[axis];
  if (count == null || !Number.isInteger(count) || count <= 0) return null;
  const range = validRange(grid, axis, count);
  if (!range) return null;
  const exact = grid.scalarCoordinatesM?.[axis];
  if (exact) {
    if (exact.length !== count || exact.some((value) => !Number.isFinite(value))) return null;
    return exact.slice(range[0], range[1] + 1);
  }
  const spacing = [grid.drM, grid.dyM, grid.dzM][axis];
  const origin = grid.scalarOriginM?.[axis];
  if (spacing == null || origin == null || !Number.isFinite(spacing) || spacing <= 0 || !Number.isFinite(origin)) return null;
  return Array.from({ length: range[1] - range[0] + 1 }, (_, index) => origin + (range[0] + index) * spacing);
}

function gridSignature(grid: GridInfo | null): { axes: [number[], number[], number[]]; staggering: string | null } | null {
  if (!grid || grid.geometryKind === "index" || grid.coordinateSystem !== "cartesian") return null;
  if (grid.haloWidth && !grid.validIndexRange) return null;
  const axes = [0, 1, 2].map((axis) => axisCoordinates(grid, axis));
  if (axes.some((axis) => axis == null)) return null;
  return {
    axes: axes as [number[], number[], number[]],
    staggering: grid.staggering ?? null,
  };
}

function sameSignature(
  left: ReturnType<typeof gridSignature>,
  right: ReturnType<typeof gridSignature>,
): boolean {
  if (!left || !right) return false;
  return left.axes.every((axis, axisIndex) =>
    axis.length === right.axes[axisIndex].length
    && axis.every((value, index) => close(value, right.axes[axisIndex][index]))
  );
}

export function alignDatasetGrids(
  leftFrames: readonly StepInfo[],
  rightFrames: readonly StepInfo[],
): DatasetGridAlignment {
  const left = gridSignature(leftFrames[0]?.grid ?? null);
  const right = gridSignature(rightFrames[0]?.grid ?? null);
  if (!left || !right) {
    return {
      status: "blocked",
      shape: null,
      maxCoordinateDeltaM: null,
      staggeringDiffers: false,
      detail: "Both datasets need complete Cartesian physical scalar-center coordinates and valid-domain metadata.",
    };
  }
  const leftStable = leftFrames.every((frame) => sameSignature(left, gridSignature(frame.grid)));
  const rightStable = rightFrames.every((frame) => sameSignature(right, gridSignature(frame.grid)));
  if (!leftStable || !rightStable) {
    return {
      status: "blocked",
      shape: null,
      maxCoordinateDeltaM: null,
      staggeringDiffers: left.staggering !== right.staggering,
      detail: "Grid coordinates change between frames within at least one dataset.",
    };
  }
  const sameShape = left.axes.every((axis, index) => axis.length === right.axes[index].length);
  if (!sameShape) {
    return {
      status: "blocked",
      shape: null,
      maxCoordinateDeltaM: null,
      staggeringDiffers: left.staggering !== right.staggering,
      detail: `Valid scalar-grid shapes differ: ${left.axes.map((axis) => axis.length).join("×")} versus ${right.axes.map((axis) => axis.length).join("×")}.`,
    };
  }
  let maxCoordinateDeltaM = 0;
  for (let axis = 0; axis < 3; axis += 1) {
    for (let index = 0; index < left.axes[axis].length; index += 1) {
      maxCoordinateDeltaM = Math.max(maxCoordinateDeltaM, Math.abs(left.axes[axis][index] - right.axes[axis][index]));
    }
  }
  if (!sameSignature(left, right)) {
    return {
      status: "blocked",
      shape: left.axes.map((axis) => axis.length) as [number, number, number],
      maxCoordinateDeltaM,
      staggeringDiffers: left.staggering !== right.staggering,
      detail: `Scalar-center coordinates differ by up to ${Number(maxCoordinateDeltaM.toPrecision(6))} m; interpolation is not yet qualified.`,
    };
  }
  return {
    status: "exact",
    shape: left.axes.map((axis) => axis.length) as [number, number, number],
    maxCoordinateDeltaM,
    staggeringDiffers: left.staggering !== right.staggering,
    detail: `Scalar-center coordinates match on a ${left.axes.map((axis) => axis.length).join("×")} valid grid.`,
  };
}

function fields(frames: readonly StepInfo[]): Set<string> {
  return new Set(frames.flatMap((frame) => frame.fields));
}

function stableUnit(field: string, frames: readonly StepInfo[]): string | null {
  const units = new Set(frames.map((frame) => frame.fieldUnits[field]).filter((unit): unit is string => Boolean(unit)));
  return units.size === 1 ? [...units][0] : null;
}

const COMPARISON_UNIT_ALIASES: Readonly<Record<string, string>> = {
  "ms-1": "m/s",
  "ms^-1": "m/s",
  "m/s": "m/s",
  "ms-2": "m/s^2",
  "ms^-2": "m/s^2",
  "m/s2": "m/s^2",
  "m/s^2": "m/s^2",
  "s-1": "1/s",
  "s^-1": "1/s",
  "1/s": "1/s",
  "kgkg-1": "kg/kg",
  "kgkg^-1": "kg/kg",
  "kg/kg": "kg/kg",
  "kgm-3": "kg/m^3",
  "kgm^-3": "kg/m^3",
  "kg/m3": "kg/m^3",
  "kg/m^3": "kg/m^3",
  "jm-2": "j/m^2",
  "jkg-1": "j/kg",
  "jkg^-1": "j/kg",
  "j/kg": "j/kg",
  "m2s-2": "m^2/s^2",
  "m^2s^-2": "m^2/s^2",
  "m2/s2": "m^2/s^2",
  "m^2/s^2": "m^2/s^2",
  "kgm-2": "kg/m^2",
  "kgm^-2": "kg/m^2",
  "kg/m2": "kg/m^2",
  "kg/m^2": "kg/m^2",
  "kgm-2s-1": "kg/m^2/s",
  "kgm^-2s^-1": "kg/m^2/s",
  "kg/m2/s": "kg/m^2/s",
  "kg/m^2/s": "kg/m^2/s",
};

/** Comparison-only unit identity. This recognizes algebraically identical
 * conventional spellings but deliberately performs no scale/offset conversion. */
export function comparisonUnitKey(unit: string): string {
  const normalized = unit
    .trim()
    .toLowerCase()
    .replace(/\u2212/g, "-")
    .replace(/\*\*/g, "^")
    .replace(/[()\s*]/g, "");
  return COMPARISON_UNIT_ALIASES[normalized] ?? normalized;
}

export function alignDatasets(
  leftSource: LocalDatasetSource,
  leftFrames: readonly StepInfo[],
  rightSource: LocalDatasetSource,
  rightFrames: readonly StepInfo[],
): DatasetAlignmentReport {
  const leftDatasetIdentity = datasetSourceIdentity(leftSource);
  const rightDatasetIdentity = datasetSourceIdentity(rightSource);
  const leftFields = fields(leftFrames);
  const rightFields = fields(rightFrames);
  const commonFields = [...leftFields].filter((field) => rightFields.has(field)).sort();
  const leftOnlyFields = [...leftFields].filter((field) => !rightFields.has(field)).sort();
  const rightOnlyFields = [...rightFields].filter((field) => !leftFields.has(field)).sort();
  const unitMismatches: DatasetUnitMismatch[] = [];
  const differenceEligibleFields = commonFields.filter((field) => {
    const leftUnit = stableUnit(field, leftFrames);
    const rightUnit = stableUnit(field, rightFrames);
    if (leftUnit && rightUnit && comparisonUnitKey(leftUnit) === comparisonUnitKey(rightUnit)) return true;
    unitMismatches.push({
      field,
      leftUnit,
      rightUnit,
      reason: leftUnit == null || rightUnit == null
        ? "A stable declared unit is missing on at least one dataset."
        : "Declared units differ; automatic conversion is not qualified.",
    });
    return false;
  });
  const grid = alignDatasetGrids(leftFrames, rightFrames);
  const time = alignDatasetTimes(leftFrames, rightFrames);
  // Motion metadata is frame-bearing. Qualifying frame zero alone would let a
  // later translating/tracked frame enter matched-time cellwise statistics.
  const motionBlocker = (time.matches.length > 0
    ? time.matches.map((match) => canCompareByCell(
      leftFrames[match.leftFrameIndex],
      rightFrames[match.rightFrameIndex],
    ))
    : [canCompareByCell(leftFrames[0], rightFrames[0])]
  ).find((verdict) => !verdict.allowed)?.blocker ?? null;
  const blockers: string[] = [];
  const warnings: string[] = [];
  if (leftFrames.length === 0 || rightFrames.length === 0) blockers.push("Both datasets need at least one complete frame.");
  if (leftDatasetIdentity === rightDatasetIdentity) blockers.push("A and B refer to the same dataset identity.");
  if (grid.status === "blocked") blockers.push(grid.detail);
  if (time.status === "blocked") blockers.push(time.detail);
  if (motionBlocker) blockers.push(motionBlocker);
  if (commonFields.length === 0) blockers.push("The datasets have no common canonical fields.");
  if (differenceEligibleFields.length === 0) blockers.push("No common field has stable matching declared units.");
  if (time.status === "partial") warnings.push(time.detail);
  if (grid.staggeringDiffers) warnings.push("Native grid-staggering labels differ; canonical fields are aligned only on their published scalar centers.");
  if (leftOnlyFields.length > 0 || rightOnlyFields.length > 0) warnings.push(`${leftOnlyFields.length} A-only and ${rightOnlyFields.length} B-only fields cannot be differenced.`);
  if (unitMismatches.length > 0) warnings.push(`${unitMismatches.length} common fields have missing or unequal declared units.`);
  const differenceAllowed = blockers.length === 0;
  const verdict: AlignmentStatus = !differenceAllowed
    ? "blocked"
    : time.status === "exact" && warnings.length === 0
      ? "exact"
      : "partial";
  return {
    schema: DATASET_ALIGNMENT_SCHEMA,
    schemaVersion: DATASET_ALIGNMENT_VERSION,
    leftDatasetIdentity,
    rightDatasetIdentity,
    verdict,
    differenceAllowed,
    commonFields,
    differenceEligibleFields,
    leftOnlyFields,
    rightOnlyFields,
    unitMismatches,
    grid,
    time,
    blockers,
    warnings,
  };
}
