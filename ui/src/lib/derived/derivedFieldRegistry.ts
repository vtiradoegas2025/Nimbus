import type { LocalDatasetSource } from "@/lib/dataset/datasetSources";
import { datasetSourceIdentity } from "@/lib/dataset/datasetSources";
import type { NpyN } from "@/lib/platform/npy";
import type { StepInfo } from "@/lib/run/runOutputs";
import {
  C_P,
  R_D,
  T_ZERO,
  dewpointCelsiusFromQv,
  equivalentPotentialTemperatureBolton,
  kelvinToCelsius,
  partialVaporPressurePa,
  satVaporPressureHPa,
} from "@/lib/science/thermo";

export const DERIVED_FIELD_REGISTRY_SCHEMA = "nimbus.studio.derived-field-registry" as const;
export const DERIVED_FIELD_REGISTRY_VERSION = 1 as const;
export const DERIVED_FIELD_PROVENANCE_SCHEMA = "nimbus.studio.derived-field-provenance" as const;
export const DERIVED_FIELD_PROVENANCE_VERSION = 1 as const;

export type BuiltinDerivedFieldId =
  | "temperature"
  | "dewpoint"
  | "relative_humidity"
  | "theta_e"
  | "horizontal_wind_speed";

export interface DerivedFieldDependency {
  field: string;
  unit: string;
}

export interface DerivedFieldParameter {
  id: string;
  label: string;
  type: "number" | "enum";
  value: number | string;
  unit: string | null;
  mutable: false;
  reason: string;
}

export interface DerivedFieldDefinition {
  schema: typeof DERIVED_FIELD_REGISTRY_SCHEMA;
  registryVersion: typeof DERIVED_FIELD_REGISTRY_VERSION;
  definitionVersion: number;
  id: BuiltinDerivedFieldId;
  label: string;
  unit: string;
  rank: 3;
  provider: "nimbus_studio_builtin";
  dependencies: readonly DerivedFieldDependency[];
  parameters: readonly DerivedFieldParameter[];
  formula: string;
  method: string;
  invalidValuePolicy: string;
}

const common = {
  schema: DERIVED_FIELD_REGISTRY_SCHEMA,
  registryVersion: DERIVED_FIELD_REGISTRY_VERSION,
  definitionVersion: 1,
  rank: 3 as const,
  provider: "nimbus_studio_builtin" as const,
  invalidValuePolicy: "Any invalid or physically inadmissible dependency cell produces NaN; no spatial filling is performed.",
};

const referencePressure: DerivedFieldParameter = {
  id: "reference_pressure_pa",
  label: "Potential-temperature reference pressure",
  type: "number",
  value: 100_000,
  unit: "Pa",
  mutable: false,
  reason: "Canonical atmospheric potential temperature uses 1000 hPa; changing it would change the field identity.",
};

export const DERIVED_FIELD_DEFINITIONS: readonly DerivedFieldDefinition[] = [
  {
    ...common,
    id: "temperature",
    label: "Temperature",
    unit: "K",
    dependencies: [{ field: "theta", unit: "K" }, { field: "p", unit: "Pa" }],
    parameters: [referencePressure],
    formula: "T = theta * (p / p_ref)^(Rd / cp)",
    method: "Poisson potential-temperature relation using Nimbus thermodynamic constants",
  },
  {
    ...common,
    id: "dewpoint",
    label: "Dewpoint",
    unit: "K",
    dependencies: [{ field: "qv", unit: "kg/kg" }, { field: "p", unit: "Pa" }],
    parameters: [{
      id: "vapor_pressure_inversion",
      label: "Vapor-pressure inversion",
      type: "enum",
      value: "magnus_liquid_water",
      unit: null,
      mutable: false,
      reason: "The field identity fixes the liquid-water Magnus inversion; alternate phase assumptions require another definition ID.",
    }],
    formula: "e = qv*p/(epsilon+qv); Td = inverse Magnus(e)",
    method: "Dry-air mixing ratio to vapor pressure, then inverse Magnus relation over liquid water",
  },
  {
    ...common,
    id: "relative_humidity",
    label: "Relative humidity",
    unit: "%",
    dependencies: [{ field: "theta", unit: "K" }, { field: "p", unit: "Pa" }, { field: "qv", unit: "kg/kg" }],
    parameters: [referencePressure, {
      id: "saturation_surface",
      label: "Saturation surface",
      type: "enum",
      value: "liquid_water",
      unit: null,
      mutable: false,
      reason: "Supersaturation is retained against liquid water; ice-relative humidity would be a distinct field.",
    }],
    formula: "RH = 100 * e(qv,p) / es_liquid(T(theta,p))",
    method: "Relative humidity over liquid water with no supersaturation clipping",
  },
  {
    ...common,
    id: "theta_e",
    label: "Equivalent potential temperature",
    unit: "K",
    dependencies: [{ field: "theta", unit: "K" }, { field: "p", unit: "Pa" }, { field: "qv", unit: "kg/kg" }],
    parameters: [referencePressure, {
      id: "formulation",
      label: "Formulation",
      type: "enum",
      value: "bolton_1980_eq_15_43",
      unit: null,
      mutable: false,
      reason: "Equivalent potential temperature has multiple definitions; this ID is pinned to Bolton (1980).",
    }],
    formula: "Bolton (1980) Eq. 15 and Eq. 43 pseudoequivalent potential temperature",
    method: "Bolton (1980) Eq. 15/43 pseudoequivalent potential temperature from T, dewpoint, pressure, and dry-air mixing ratio",
  },
  {
    ...common,
    id: "horizontal_wind_speed",
    label: "Horizontal wind speed",
    unit: "m/s",
    dependencies: [{ field: "u", unit: "m/s" }, { field: "v", unit: "m/s" }],
    parameters: [],
    formula: "sqrt(u^2 + v^2)",
    method: "Euclidean magnitude of canonical cell-centered horizontal wind components",
  },
] as const;

for (const definition of DERIVED_FIELD_DEFINITIONS) {
  for (const dependency of definition.dependencies) Object.freeze(dependency);
  for (const parameter of definition.parameters) Object.freeze(parameter);
  Object.freeze(definition.dependencies);
  Object.freeze(definition.parameters);
  Object.freeze(definition);
}
Object.freeze(DERIVED_FIELD_DEFINITIONS);

export interface DerivedFieldAssessment {
  definition: DerivedFieldDefinition;
  status: "available" | "native_precedence" | "blocked";
  missingDependencies: string[];
  incompatibleUnits: Array<{ field: string; expected: string; actual: string | null }>;
}

function normalizeUnit(unit: string): string {
  return unit.trim().toLowerCase().replaceAll(" ", "").replaceAll("**", "^");
}

function unitsMatch(actual: string | undefined, expected: string): boolean {
  if (!actual?.trim()) return false;
  const left = normalizeUnit(actual);
  const right = normalizeUnit(expected);
  const aliases: Readonly<Record<string, readonly string[]>> = {
    "m/s": ["ms^-1", "ms-1"],
    "kg/kg": ["kgkg^-1", "kgkg-1"],
  };
  return left === right || aliases[right]?.includes(left) === true;
}

export function derivedFieldDefinition(fieldName: string): DerivedFieldDefinition | null {
  return DERIVED_FIELD_DEFINITIONS.find(({ id }) => id === fieldName) ?? null;
}

export function assessDerivedFields(
  nativeFields: ReadonlySet<string>,
  fieldUnits: Readonly<Record<string, string>>,
): DerivedFieldAssessment[] {
  const availableFields = new Set(nativeFields);
  const availableUnits: Record<string, string> = { ...fieldUnits };
  const assessments: DerivedFieldAssessment[] = [];
  for (const definition of DERIVED_FIELD_DEFINITIONS) {
    if (nativeFields.has(definition.id)) {
      assessments.push({ definition, status: "native_precedence", missingDependencies: [], incompatibleUnits: [] });
      continue;
    }
    const missingDependencies = definition.dependencies
      .filter(({ field }) => !availableFields.has(field))
      .map(({ field }) => field);
    const incompatibleUnits = definition.dependencies.flatMap(({ field, unit }) =>
      availableFields.has(field) && !unitsMatch(availableUnits[field], unit)
        ? [{ field, expected: unit, actual: availableUnits[field] ?? null }]
        : []
    );
    const status = missingDependencies.length === 0 && incompatibleUnits.length === 0 ? "available" : "blocked";
    assessments.push({ definition, status, missingDependencies, incompatibleUnits });
    if (status === "available") {
      availableFields.add(definition.id);
      availableUnits[definition.id] = definition.unit;
    }
  }
  return assessments;
}

export function augmentStepWithDerivedFields(step: StepInfo): StepInfo {
  const origins = { ...Object.fromEntries(step.fields.map((field) => [field, "native"] as const)), ...step.fieldOrigins };
  const assessments = assessDerivedFields(
    new Set(Object.entries(origins).filter(([, origin]) => origin === "native").map(([field]) => field)),
    step.fieldUnits,
  );
  const available = assessments.filter(({ status }) => status === "available");
  return {
    ...step,
    fields: [...step.fields, ...available.map(({ definition }) => definition.id).filter((field) => !step.fields.includes(field))],
    fieldUnits: { ...step.fieldUnits, ...Object.fromEntries(available.map(({ definition }) => [definition.id, definition.unit])) },
    fieldOrigins: { ...origins, ...Object.fromEntries(available.map(({ definition }) => [definition.id, "studio_derived" as const])) },
  };
}

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((length, axis) => length === right[axis]);
}

function fixedNumber(definition: DerivedFieldDefinition, id: string): number {
  const value = definition.parameters.find((parameter) => parameter.id === id)?.value;
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`Derived field '${definition.id}' has invalid fixed parameter '${id}'`);
  return value;
}

export function evaluateDerivedField(fieldName: string, inputs: Readonly<Record<string, NpyN>>): NpyN {
  const definition = derivedFieldDefinition(fieldName);
  if (!definition) throw new Error(`Unknown derived field '${fieldName}'`);
  const reference = inputs[definition.dependencies[0].field];
  if (!reference) throw new Error(`Derived field '${fieldName}' is missing '${definition.dependencies[0].field}'`);
  for (const { field } of definition.dependencies) {
    const input = inputs[field];
    if (!input) throw new Error(`Derived field '${fieldName}' is missing '${field}'`);
    if (!sameShape(reference.shape, input.shape)) throw new Error(`Derived field '${fieldName}' dependency shape mismatch for '${field}'`);
  }
  const output = new Float32Array(reference.data.length);
  const theta = inputs.theta?.data; const pressure = inputs.p?.data; const qv = inputs.qv?.data;
  const u = inputs.u?.data; const v = inputs.v?.data;
  const pRef = definition.parameters.some(({ id }) => id === "reference_pressure_pa")
    ? fixedNumber(definition, "reference_pressure_pa")
    : 100_000;
  for (let index = 0; index < output.length; index += 1) {
    if (fieldName === "horizontal_wind_speed") {
      const uValue = u?.[index] ?? NaN; const vValue = v?.[index] ?? NaN;
      output[index] = Number.isFinite(uValue) && Number.isFinite(vValue) ? Math.hypot(uValue, vValue) : NaN;
      continue;
    }
    const pPa = pressure?.[index] ?? NaN; const thetaK = theta?.[index] ?? NaN; const vapor = qv?.[index] ?? NaN;
    if (!(pPa > 0) || !Number.isFinite(pPa)) { output[index] = NaN; continue; }
    const temperatureK = thetaK > 0 && Number.isFinite(thetaK) ? thetaK * Math.pow(pPa / pRef, R_D / C_P) : NaN;
    if (fieldName === "temperature") output[index] = temperatureK;
    else if (fieldName === "dewpoint") {
      const dewpointC = vapor >= 0 && Number.isFinite(vapor) ? dewpointCelsiusFromQv(vapor, pPa) : NaN;
      output[index] = Number.isFinite(dewpointC) ? dewpointC + T_ZERO : NaN;
    } else if (fieldName === "relative_humidity") {
      if (!Number.isFinite(temperatureK) || vapor < 0 || !Number.isFinite(vapor)) { output[index] = NaN; continue; }
      const saturationPa = satVaporPressureHPa(kelvinToCelsius(temperatureK)) * 100;
      const vaporPressurePa = partialVaporPressurePa(vapor, pPa);
      output[index] = saturationPa > 0 && Number.isFinite(saturationPa) ? 100 * vaporPressurePa / saturationPa : NaN;
    } else output[index] = equivalentPotentialTemperatureBolton(thetaK, pPa, vapor);
  }
  return { shape: [...reference.shape], data: output };
}

export function derivedFieldCacheIdentity(fieldName: string): string {
  const definition = derivedFieldDefinition(fieldName);
  if (!definition) return fieldName;
  return JSON.stringify([
    DERIVED_FIELD_REGISTRY_SCHEMA,
    DERIVED_FIELD_REGISTRY_VERSION,
    definition.id,
    definition.definitionVersion,
    definition.parameters.map(({ id, value }) => [id, value]),
  ]);
}

export interface DerivedFieldProvenance {
  schema: typeof DERIVED_FIELD_PROVENANCE_SCHEMA;
  schemaVersion: typeof DERIVED_FIELD_PROVENANCE_VERSION;
  datasetIdentity: string;
  sourceKind: LocalDatasetSource["kind"];
  frameIndex: number;
  simulationTimeS: number | null;
  publicationGeneration: string | null;
  definition: DerivedFieldDefinition;
  cacheIdentity: string;
  dependencyIdentities: string[];
}

export function buildDerivedFieldProvenance(
  dataset: LocalDatasetSource,
  step: StepInfo,
  fieldName: string,
): DerivedFieldProvenance {
  const definition = derivedFieldDefinition(fieldName);
  if (!definition) throw new Error(`Unknown derived field '${fieldName}'`);
  const identity = datasetSourceIdentity(dataset);
  return {
    schema: DERIVED_FIELD_PROVENANCE_SCHEMA,
    schemaVersion: DERIVED_FIELD_PROVENANCE_VERSION,
    datasetIdentity: identity,
    sourceKind: dataset.kind,
    frameIndex: step.index,
    simulationTimeS: step.simulationTimeS,
    publicationGeneration: step.publicationGeneration,
    definition,
    cacheIdentity: derivedFieldCacheIdentity(fieldName),
    dependencyIdentities: definition.dependencies.map(({ field }) =>
      JSON.stringify([identity, step.publicationGeneration, step.index, field])
    ),
  };
}
