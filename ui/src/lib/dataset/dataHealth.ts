import { datasetSourceIdentity, type LocalDatasetSource } from "./datasetSources";
import { analyzeDatasetTime, type DatasetTimeHealth } from "./datasetTimeHealth";
import type { NpyN } from "@/lib/platform/npy";
import type { StepInfo } from "@/lib/run/runOutputs";

export const DATA_HEALTH_SCHEMA = "nimbus.studio.data-health" as const;
export const DATA_HEALTH_VERSION = 1 as const;

export interface FieldValueHealth {
  elementCount: number;
  finiteCount: number;
  nanCount: number;
  positiveInfinityCount: number;
  negativeInfinityCount: number;
  minimum: number | null;
  maximum: number | null;
  mean: number | null;
  constantFiniteField: boolean;
  physicalRule: string | null;
  physicalViolationCount: number;
}

export interface FieldFrameHealth extends FieldValueHealth {
  stepIndex: number;
  simulationTimeS: number | null;
  shape: number[];
}

export interface DatasetStructureHealth {
  frameCount: number;
  physicalFrameCount: number;
  indexFrameCount: number;
  missingGeometryFrameCount: number;
  exactCoordinateFrameCount: number;
  unionFieldCount: number;
  sharedFieldCount: number;
  varyingFieldCount: number;
  fieldsMissingUnits: string[];
  unitCompatibilityIssues: Array<{ family: string; fields: string[]; units: string[] }>;
  time: DatasetTimeHealth;
}

interface PhysicalRule {
  description: string;
  valid: (value: number) => boolean;
}

const POSITIVE_FIELDS = new Set(["p", "pressure", "rho", "density", "theta", "temperature", "theta_e"]);
const NONNEGATIVE_FIELDS = new Set([
  "qv", "qc", "qr", "qi", "qs", "qg", "cloud_condensate", "cloud_liquid_condensate",
  "cloud_frozen_condensate", "relative_humidity", "accumulated_rainfall", "precip_rate",
  "precipitable_water", "liquid_water_path", "cloud_water_path", "cape", "lcl", "lfc",
]);

export function physicalRuleForField(fieldName: string): PhysicalRule | null {
  if (POSITIVE_FIELDS.has(fieldName)) return { description: `${fieldName} must be finite and greater than zero`, valid: (value) => value > 0 };
  if (NONNEGATIVE_FIELDS.has(fieldName)) return { description: `${fieldName} must be finite and nonnegative`, valid: (value) => value >= 0 };
  return null;
}

export interface DatasetFieldHealthReport extends FieldValueHealth {
  schema: typeof DATA_HEALTH_SCHEMA;
  schemaVersion: typeof DATA_HEALTH_VERSION;
  generatedAt: string;
  dataset: LocalDatasetSource;
  datasetIdentity: string;
  fieldName: string;
  unit: string;
  requestedScope: "current_frame" | "all_available_frames";
  requestedFrameCount: number;
  scannedFrameCount: number;
  missingFieldFrameCount: number;
  structure: DatasetStructureHealth;
  frames: FieldFrameHealth[];
}

export function analyzeFieldValues(field: NpyN, fieldName = ""): FieldValueHealth {
  let finiteCount = 0; let nanCount = 0; let positiveInfinityCount = 0; let negativeInfinityCount = 0;
  let physicalViolationCount = 0;
  const physicalRule = physicalRuleForField(fieldName);
  let minimum = Number.POSITIVE_INFINITY; let maximum = Number.NEGATIVE_INFINITY; let sum = 0; let compensation = 0;
  for (const value of field.data) {
    if (Number.isNaN(value)) { nanCount += 1; continue; }
    if (value === Number.POSITIVE_INFINITY) { positiveInfinityCount += 1; continue; }
    if (value === Number.NEGATIVE_INFINITY) { negativeInfinityCount += 1; continue; }
    if (!Number.isFinite(value)) { nanCount += 1; continue; }
    if (physicalRule && !physicalRule.valid(value)) physicalViolationCount += 1;
    finiteCount += 1; minimum = Math.min(minimum, value); maximum = Math.max(maximum, value);
    const adjusted = value - compensation; const next = sum + adjusted; compensation = (next - sum) - adjusted; sum = next;
  }
  return {
    elementCount: field.data.length,
    finiteCount,
    nanCount,
    positiveInfinityCount,
    negativeInfinityCount,
    minimum: finiteCount > 0 ? minimum : null,
    maximum: finiteCount > 0 ? maximum : null,
    mean: finiteCount > 0 ? sum / finiteCount : null,
    constantFiniteField: finiteCount > 0 && minimum === maximum,
    physicalRule: physicalRule?.description ?? null,
    physicalViolationCount,
  };
}

function normalizedUnit(unit: string): string {
  return unit.trim().toLowerCase().replaceAll(" ", "").replaceAll("**", "^");
}

function unitCompatibilityIssues(steps: readonly StepInfo[]): DatasetStructureHealth["unitCompatibilityIssues"] {
  const families: Array<{ family: string; fields: string[] }> = [
    { family: "wind components", fields: ["u", "v", "w"] },
    { family: "thermodynamic temperatures", fields: ["theta", "theta_prime", "temperature", "dewpoint", "theta_e"] },
    { family: "water mixing ratios", fields: ["qv", "qc", "qr", "qi", "qs", "qg"] },
  ];
  return families.flatMap(({ family, fields }) => {
    const present = fields.filter((field) => steps.some((step) => step.fields.includes(field)));
    const units = [...new Set(steps.flatMap((step) => present.map((field) => step.fieldUnits[field]?.trim()).filter((unit): unit is string => Boolean(unit))).map(normalizedUnit))];
    return present.length > 1 && units.length > 1 ? [{ family, fields: present, units }] : [];
  });
}

export function analyzeDatasetStructure(steps: readonly StepInfo[]): DatasetStructureHealth {
  const union = new Set(steps.flatMap((step) => step.fields));
  const shared = steps.length === 0 ? new Set<string>() : new Set(steps[0].fields.filter((field) => steps.every((step) => step.fields.includes(field))));
  const missingUnits = [...union].filter((field) => steps.some((step) => step.fields.includes(field) && !(step.fieldUnits[field]?.trim()))).sort();
  const exactCoordinateFrameCount = steps.filter((step) => {
    const coords = step.grid?.scalarCoordinatesM;
    const expectedLengths = [step.grid?.nr, step.grid?.nth, step.grid?.nz];
    return coords?.length === 3 && coords.every((axis, axisIndex) =>
      expectedLengths[axisIndex] != null
      && axis.length === expectedLengths[axisIndex]
      && axis.length > 0
      && axis.every((value, index) => Number.isFinite(value) && (index === 0 || value > axis[index - 1]))
    );
  }).length;
  return {
    frameCount: steps.length,
    physicalFrameCount: steps.filter((step) => step.grid?.geometryKind === "physical").length,
    indexFrameCount: steps.filter((step) => step.grid?.geometryKind === "index").length,
    missingGeometryFrameCount: steps.filter((step) => !step.grid).length,
    exactCoordinateFrameCount,
    unionFieldCount: union.size,
    sharedFieldCount: shared.size,
    varyingFieldCount: union.size - shared.size,
    fieldsMissingUnits: missingUnits,
    unitCompatibilityIssues: unitCompatibilityIssues(steps),
    time: analyzeDatasetTime(steps),
  };
}

export function createDatasetFieldHealthReport(args: {
  dataset: LocalDatasetSource;
  fieldName: string;
  unit: string;
  requestedScope: DatasetFieldHealthReport["requestedScope"];
  requestedFrameCount: number;
  missingFieldFrameCount: number;
  structure: DatasetStructureHealth;
  frames: readonly FieldFrameHealth[];
}): DatasetFieldHealthReport {
  let totalElements = 0; let finiteCount = 0; let nanCount = 0; let positiveInfinityCount = 0; let negativeInfinityCount = 0;
  let physicalViolationCount = 0;
  let minimum = Number.POSITIVE_INFINITY; let maximum = Number.NEGATIVE_INFINITY;
  let weightedSum = 0; let weightedCompensation = 0;
  for (const frame of args.frames) {
    totalElements += frame.elementCount; finiteCount += frame.finiteCount; nanCount += frame.nanCount;
    positiveInfinityCount += frame.positiveInfinityCount; negativeInfinityCount += frame.negativeInfinityCount;
    physicalViolationCount += frame.physicalViolationCount;
    if (frame.minimum != null) minimum = Math.min(minimum, frame.minimum);
    if (frame.maximum != null) maximum = Math.max(maximum, frame.maximum);
    if (frame.mean != null) {
      const adjusted = frame.mean * frame.finiteCount - weightedCompensation;
      const next = weightedSum + adjusted;
      weightedCompensation = (next - weightedSum) - adjusted;
      weightedSum = next;
    }
  }
  return {
    schema: DATA_HEALTH_SCHEMA, schemaVersion: DATA_HEALTH_VERSION, generatedAt: new Date().toISOString(),
    dataset: args.dataset, datasetIdentity: datasetSourceIdentity(args.dataset), fieldName: args.fieldName,
    unit: args.unit, requestedScope: args.requestedScope, requestedFrameCount: args.requestedFrameCount,
    scannedFrameCount: args.frames.length, missingFieldFrameCount: args.missingFieldFrameCount,
    structure: args.structure, frames: [...args.frames], elementCount: totalElements, finiteCount, nanCount,
    positiveInfinityCount, negativeInfinityCount, minimum: finiteCount > 0 ? minimum : null,
    maximum: finiteCount > 0 ? maximum : null, mean: finiteCount > 0 ? weightedSum / finiteCount : null,
    constantFiniteField: finiteCount > 0 && minimum === maximum,
    physicalRule: args.frames.find(({ physicalRule }) => physicalRule != null)?.physicalRule ?? null,
    physicalViolationCount,
  };
}

export function serializeDatasetFieldHealth(report: DatasetFieldHealthReport): string {
  return `${JSON.stringify(report, null, 2)}\n`;
}
