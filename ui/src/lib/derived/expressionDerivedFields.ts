import { datasetSourceIdentity, type LocalDatasetSource } from "@/lib/dataset/datasetSources";
import { compileExpression, type CompiledExpression } from "./exprEval";
import type { NpyN } from "@/lib/platform/npy";
import type { StepInfo } from "@/lib/run/runOutputs";

export const EXPRESSION_DERIVED_FIELD_SCHEMA = "nimbus.studio.expression-derived-field" as const;
export const EXPRESSION_DERIVED_FIELD_SCHEMA_VERSION = 1 as const;
export const EXPRESSION_DERIVED_FIELD_SET_SCHEMA = "nimbus.studio.expression-derived-field-set" as const;
export const EXPRESSION_DERIVED_FIELD_PROVENANCE_SCHEMA = "nimbus.studio.expression-derived-field-provenance" as const;
export const MAX_EXPRESSION_DERIVED_FIELDS = 24;
export const MAX_EXPRESSION_ELEMENTS = 5_000_000;

const MAX_DEPENDENCIES = 16;
const MAX_PARAMETERS = 16;
const MAX_FORMULA_LENGTH = 2_000;
const IDENTIFIER = /^[a-z][a-z0-9_]{0,63}$/;
const ALLOWED_DEFINITION_KEYS = new Set([
  "schema", "schemaVersion", "definitionVersion", "id", "label", "unit", "rank",
  "provider", "dependencies", "parameters", "formula", "invalidValuePolicy",
]);
const RESERVED_IDENTIFIERS = new Set([
  "abs", "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh", "ceil", "cos",
  "cosh", "exp", "floor", "hypot", "if", "length", "ln", "log", "log10", "max", "min",
  "pow", "random", "round", "sign", "sin", "sinh", "sqrt", "tan", "tanh", "trunc",
  "true", "false", "null", "undefined", "nan", "infinity", "e", "pi", "and", "or", "not",
  "constructor", "prototype", "__proto__", "this", "new", "function", "return", "window",
  "document", "globalThis", "eval",
]);

export interface ExpressionDerivedFieldParameter {
  id: string;
  value: number;
}

export interface ExpressionDerivedFieldDefinition {
  schema: typeof EXPRESSION_DERIVED_FIELD_SCHEMA;
  schemaVersion: typeof EXPRESSION_DERIVED_FIELD_SCHEMA_VERSION;
  definitionVersion: 1;
  id: string;
  label: string;
  unit: string;
  rank: 3;
  provider: "nimbus_studio_user_expression";
  dependencies: readonly string[];
  parameters: readonly ExpressionDerivedFieldParameter[];
  formula: string;
  invalidValuePolicy: "Any nonfinite dependency produces NaN; nonfinite expression results remain NaN; no filling is performed.";
}

export interface ExpressionDerivedFieldDraft {
  id: string;
  label: string;
  unit: string;
  dependencies: readonly string[];
  parameters?: readonly ExpressionDerivedFieldParameter[];
  formula: string;
}

export interface ExpressionDefinitionValidationOptions {
  nativeFields?: ReadonlySet<string>;
  reservedFieldIds?: ReadonlySet<string>;
  availableFields?: ReadonlySet<string>;
  existingDefinitionIds?: ReadonlySet<string>;
  replacingId?: string;
}

export interface CompiledExpressionDerivedField {
  definition: ExpressionDerivedFieldDefinition;
  compiled: CompiledExpression;
  parameterValues: Readonly<Record<string, number>>;
}

export interface ExpressionDerivedFieldLoadResult {
  definitions: ExpressionDerivedFieldDefinition[];
  warnings: string[];
}

export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requireString(value: unknown, label: string, maxLength: number): string {
  if (typeof value !== "string") throw new Error(`${label} must be a string`);
  const normalized = value.trim();
  if (!normalized) throw new Error(`${label} must not be empty`);
  if (normalized.length > maxLength) throw new Error(`${label} exceeds ${maxLength} characters`);
  return normalized;
}

function requireIdentifier(value: unknown, label: string): string {
  const identifier = requireString(value, label, 64);
  if (!IDENTIFIER.test(identifier)) {
    throw new Error(`${label} '${identifier}' must start with a lowercase letter and contain only lowercase letters, digits, or underscores`);
  }
  if (identifier.startsWith("__") || RESERVED_IDENTIFIERS.has(identifier.toLowerCase())) {
    throw new Error(`${label} '${identifier}' is reserved by the expression evaluator`);
  }
  return identifier;
}

function exactKeys(value: Record<string, unknown>, allowed: ReadonlySet<string>, label: string): void {
  const unknown = Object.keys(value).filter((key) => !allowed.has(key));
  if (unknown.length) throw new Error(`${label} contains unsupported ${unknown.length === 1 ? "property" : "properties"}: ${unknown.join(", ")}`);
}

function normalizeDependencies(value: unknown): string[] {
  if (!Array.isArray(value) || value.length === 0) throw new Error("dependencies must contain at least one field identifier");
  if (value.length > MAX_DEPENDENCIES) throw new Error(`dependencies exceed the ${MAX_DEPENDENCIES}-field limit`);
  const dependencies = value.map((entry, index) => requireIdentifier(entry, `dependency ${index + 1}`));
  if (new Set(dependencies).size !== dependencies.length) throw new Error("dependencies must not contain duplicates");
  return dependencies.sort();
}

function normalizeParameters(value: unknown): ExpressionDerivedFieldParameter[] {
  if (value == null) return [];
  if (!Array.isArray(value)) throw new Error("parameters must be an array");
  if (value.length > MAX_PARAMETERS) throw new Error(`parameters exceed the ${MAX_PARAMETERS}-parameter limit`);
  const parameters = value.map((entry, index) => {
    if (!isRecord(entry)) throw new Error(`parameter ${index + 1} must be an object`);
    exactKeys(entry, new Set(["id", "value"]), `parameter ${index + 1}`);
    const id = requireIdentifier(entry.id, `parameter ${index + 1} id`);
    if (typeof entry.value !== "number" || !Number.isFinite(entry.value)) {
      throw new Error(`parameter '${id}' must have a finite numeric value`);
    }
    return { id, value: entry.value };
  });
  if (new Set(parameters.map(({ id }) => id)).size !== parameters.length) throw new Error("parameters must not contain duplicate identifiers");
  return parameters.sort((left, right) => left.id.localeCompare(right.id));
}

function validateDefinitionRelationships(
  definition: ExpressionDerivedFieldDefinition,
  options: ExpressionDefinitionValidationOptions,
): CompiledExpression {
  if (options.nativeFields?.has(definition.id)) throw new Error(`field id '${definition.id}' collides with a native dataset field; native fields always win`);
  if (options.reservedFieldIds?.has(definition.id)) throw new Error(`field id '${definition.id}' collides with an existing dataset or registry field`);
  if (options.existingDefinitionIds?.has(definition.id) && options.replacingId !== definition.id) {
    throw new Error(`field id '${definition.id}' already exists in this dataset`);
  }
  const parameterIds = new Set(definition.parameters.map(({ id }) => id));
  const overlap = definition.dependencies.find((dependency) => parameterIds.has(dependency));
  if (overlap) throw new Error(`identifier '${overlap}' cannot be both a dependency and a parameter`);
  if (definition.dependencies.includes(definition.id) || parameterIds.has(definition.id)) {
    throw new Error(`field '${definition.id}' cannot depend on or parameterize itself`);
  }
  if (options.availableFields) {
    const missing = definition.dependencies.filter((dependency) => !options.availableFields?.has(dependency));
    if (missing.length) throw new Error(`expression dependencies are not available in this dataset: ${missing.join(", ")}`);
  }
  if (/@step\s*:|@/.test(definition.formula)) throw new Error("cross-frame references are not allowed in registry expressions");
  const declarations = [...definition.dependencies, ...definition.parameters.map(({ id }) => id)];
  const compiled = compileExpression(definition.formula, declarations);
  if (compiled.frameRefs.length) throw new Error("cross-frame references are not allowed in registry expressions");
  return compiled;
}

export function createExpressionDerivedField(
  draft: ExpressionDerivedFieldDraft,
  options: ExpressionDefinitionValidationOptions = {},
): ExpressionDerivedFieldDefinition {
  if (!isRecord(draft)) throw new Error("expression-derived field draft must be an object");
  const definition: ExpressionDerivedFieldDefinition = {
    schema: EXPRESSION_DERIVED_FIELD_SCHEMA,
    schemaVersion: EXPRESSION_DERIVED_FIELD_SCHEMA_VERSION,
    definitionVersion: 1,
    id: requireIdentifier(draft.id, "field id"),
    label: requireString(draft.label, "label", 120),
    unit: requireString(draft.unit, "unit", 48),
    rank: 3,
    provider: "nimbus_studio_user_expression",
    dependencies: normalizeDependencies(draft.dependencies),
    parameters: normalizeParameters(draft.parameters),
    formula: requireString(draft.formula, "formula", MAX_FORMULA_LENGTH),
    invalidValuePolicy: "Any nonfinite dependency produces NaN; nonfinite expression results remain NaN; no filling is performed.",
  };
  validateDefinitionRelationships(definition, options);
  for (const parameter of definition.parameters) Object.freeze(parameter);
  Object.freeze(definition.dependencies);
  Object.freeze(definition.parameters);
  return Object.freeze(definition);
}

export function parseExpressionDerivedField(
  value: unknown,
  options: ExpressionDefinitionValidationOptions = {},
): ExpressionDerivedFieldDefinition {
  if (!isRecord(value)) throw new Error("expression-derived field definition must be an object");
  exactKeys(value, ALLOWED_DEFINITION_KEYS, "expression-derived field definition");
  if (value.schema !== EXPRESSION_DERIVED_FIELD_SCHEMA || value.schemaVersion !== EXPRESSION_DERIVED_FIELD_SCHEMA_VERSION) {
    throw new Error("unsupported expression-derived field schema or version");
  }
  if (value.definitionVersion !== 1 || value.rank !== 3 || value.provider !== "nimbus_studio_user_expression") {
    throw new Error("unsupported expression-derived field contract");
  }
  if (value.invalidValuePolicy !== "Any nonfinite dependency produces NaN; nonfinite expression results remain NaN; no filling is performed.") {
    throw new Error("unsupported expression-derived field invalid-value policy");
  }
  return createExpressionDerivedField({
    id: value.id as string,
    label: value.label as string,
    unit: value.unit as string,
    dependencies: value.dependencies as string[],
    parameters: value.parameters as ExpressionDerivedFieldParameter[],
    formula: value.formula as string,
  }, options);
}

export function compileExpressionDerivedField(
  definition: ExpressionDerivedFieldDefinition,
  options: ExpressionDefinitionValidationOptions = {},
): CompiledExpressionDerivedField {
  const canonical = parseExpressionDerivedField(definition, options);
  return {
    definition: canonical,
    compiled: validateDefinitionRelationships(canonical, options),
    parameterValues: Object.freeze(Object.fromEntries(canonical.parameters.map(({ id, value }) => [id, value]))),
  };
}

function elementCount(shape: readonly number[]): number {
  if (shape.length === 0 || shape.some((length) => !Number.isSafeInteger(length) || length <= 0)) {
    throw new Error("expression inputs must have a non-empty shape of positive integer lengths");
  }
  const count = shape.reduce((product, length) => product * length, 1);
  if (!Number.isSafeInteger(count)) throw new Error("expression input shape exceeds the safe element-count range");
  return count;
}

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((length, axis) => length === right[axis]);
}

export function evaluateExpressionDerivedField(
  definitionOrCompiled: ExpressionDerivedFieldDefinition | CompiledExpressionDerivedField,
  inputs: Readonly<Record<string, NpyN>>,
  options: { maxElements?: number } = {},
): NpyN {
  const prepared = "compiled" in definitionOrCompiled
    ? definitionOrCompiled
    : compileExpressionDerivedField(definitionOrCompiled);
  const { definition, compiled, parameterValues } = prepared;
  const reference = inputs[definition.dependencies[0]];
  if (!reference) throw new Error(`expression field '${definition.id}' is missing dependency '${definition.dependencies[0]}'`);
  const count = elementCount(reference.shape);
  if (reference.data.length !== count) throw new Error(`dependency '${definition.dependencies[0]}' data length does not match its declared shape`);
  const requestedCap = options.maxElements ?? MAX_EXPRESSION_ELEMENTS;
  if (!Number.isSafeInteger(requestedCap) || requestedCap <= 0) throw new Error("maxElements must be a positive safe integer");
  const cap = Math.min(requestedCap, MAX_EXPRESSION_ELEMENTS);
  if (count > cap) throw new Error(`expression input too large (${count} elements > ${cap}); slice or reduce before evaluation`);
  const arrays: Float32Array[] = [];
  for (const dependency of definition.dependencies) {
    const input = inputs[dependency];
    if (!input) throw new Error(`expression field '${definition.id}' is missing dependency '${dependency}'`);
    if (!sameShape(reference.shape, input.shape) || input.data.length !== count) {
      throw new Error(`dependency '${dependency}' shape ${input.shape.join("×")} does not exactly match ${reference.shape.join("×")}`);
    }
    arrays.push(input.data);
  }
  const dependencyIndex = new Map(definition.dependencies.map((id, index) => [id, index]));
  const output = new Float32Array(count);
  const args = new Array<number>(compiled.paramOrder.length);
  for (let element = 0; element < count; element += 1) {
    let invalid = false;
    for (let index = 0; index < compiled.paramOrder.length; index += 1) {
      const id = compiled.paramOrder[index];
      const arrayIndex = dependencyIndex.get(id);
      const value = arrayIndex == null ? parameterValues[id] : arrays[arrayIndex][element];
      args[index] = value;
      if (arrayIndex != null && !Number.isFinite(value)) invalid = true;
    }
    if (invalid) { output[element] = NaN; continue; }
    const value = compiled.fn(...args);
    const floatValue = typeof value === "number" ? Math.fround(value) : NaN;
    output[element] = Number.isFinite(floatValue) ? floatValue : NaN;
  }
  return { shape: [...reference.shape], data: output };
}

export function expressionDerivedFieldDefinitionIdentity(definition: ExpressionDerivedFieldDefinition): string {
  const canonical = parseExpressionDerivedField(definition);
  return JSON.stringify([
    canonical.schema, canonical.schemaVersion, canonical.definitionVersion, canonical.id,
    canonical.unit, canonical.dependencies, canonical.formula,
    canonical.parameters.map(({ id, value }) => [id, value]), canonical.invalidValuePolicy,
  ]);
}

export function expressionDerivedFieldCacheIdentity(definition: ExpressionDerivedFieldDefinition): string {
  return expressionDerivedFieldDefinitionIdentity(definition);
}

function fnv1a(value: string): string {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

export function expressionDerivedFieldStorageKey(datasetIdentity: string): string {
  if (!datasetIdentity) throw new Error("dataset identity must not be empty");
  return `nimbus.studio.expression-derived-fields.v1.${fnv1a(datasetIdentity)}`;
}

export function saveExpressionDerivedFields(
  storage: StorageLike,
  datasetIdentity: string,
  definitions: readonly ExpressionDerivedFieldDefinition[],
  options: Pick<ExpressionDefinitionValidationOptions, "nativeFields" | "reservedFieldIds" | "availableFields"> = {},
): void {
  if (definitions.length > MAX_EXPRESSION_DERIVED_FIELDS) throw new Error(`a dataset may store at most ${MAX_EXPRESSION_DERIVED_FIELDS} expression-derived fields`);
  const ids = new Set<string>();
  const canonical = definitions.map((definition) => {
    const parsed = parseExpressionDerivedField(definition, { ...options, existingDefinitionIds: ids });
    ids.add(parsed.id);
    return parsed;
  }).sort((left, right) => left.id.localeCompare(right.id));
  storage.setItem(expressionDerivedFieldStorageKey(datasetIdentity), JSON.stringify({
    schema: EXPRESSION_DERIVED_FIELD_SET_SCHEMA,
    schemaVersion: 1,
    datasetIdentity,
    definitions: canonical,
  }));
}

export function loadExpressionDerivedFields(
  storage: StorageLike,
  datasetIdentity: string,
  options: Pick<ExpressionDefinitionValidationOptions, "nativeFields" | "reservedFieldIds" | "availableFields"> = {},
): ExpressionDerivedFieldLoadResult {
  let raw: string | null;
  try { raw = storage.getItem(expressionDerivedFieldStorageKey(datasetIdentity)); }
  catch (error) { return { definitions: [], warnings: [`Expression storage is unavailable: ${error instanceof Error ? error.message : String(error)}`] }; }
  if (raw == null) return { definitions: [], warnings: [] };
  try {
    const value: unknown = JSON.parse(raw);
    if (!isRecord(value) || value.schema !== EXPRESSION_DERIVED_FIELD_SET_SCHEMA || value.schemaVersion !== 1 || value.datasetIdentity !== datasetIdentity || !Array.isArray(value.definitions)) {
      throw new Error("stored expression set has an invalid schema, version, or dataset identity");
    }
    exactKeys(value, new Set(["schema", "schemaVersion", "datasetIdentity", "definitions"]), "stored expression set");
    if (value.definitions.length > MAX_EXPRESSION_DERIVED_FIELDS) throw new Error(`stored expression set exceeds the ${MAX_EXPRESSION_DERIVED_FIELDS}-definition limit`);
    const ids = new Set<string>();
    const definitions = value.definitions.map((definition) => {
      const parsed = parseExpressionDerivedField(definition, { ...options, existingDefinitionIds: ids });
      ids.add(parsed.id);
      return parsed;
    });
    return { definitions, warnings: [] };
  } catch (error) {
    return { definitions: [], warnings: [`Stored expressions were rejected: ${error instanceof Error ? error.message : String(error)}`] };
  }
}

export function clearExpressionDerivedFields(storage: StorageLike, datasetIdentity: string): void {
  storage.removeItem(expressionDerivedFieldStorageKey(datasetIdentity));
}

export interface ExpressionDerivedFieldProvenance {
  schema: typeof EXPRESSION_DERIVED_FIELD_PROVENANCE_SCHEMA;
  schemaVersion: 1;
  datasetIdentity: string;
  sourceKind: LocalDatasetSource["kind"];
  frameIndex: number;
  simulationTimeS: number | null;
  publicationGeneration: string | null;
  definition: ExpressionDerivedFieldDefinition;
  cacheIdentity: string;
  dependencyIdentities: string[];
}

export function buildExpressionDerivedFieldProvenance(
  dataset: LocalDatasetSource,
  step: StepInfo,
  definition: ExpressionDerivedFieldDefinition,
): ExpressionDerivedFieldProvenance {
  const canonical = parseExpressionDerivedField(definition);
  const identity = datasetSourceIdentity(dataset);
  return {
    schema: EXPRESSION_DERIVED_FIELD_PROVENANCE_SCHEMA,
    schemaVersion: 1,
    datasetIdentity: identity,
    sourceKind: dataset.kind,
    frameIndex: step.index,
    simulationTimeS: step.simulationTimeS,
    publicationGeneration: step.publicationGeneration,
    definition: canonical,
    cacheIdentity: expressionDerivedFieldCacheIdentity(canonical),
    dependencyIdentities: canonical.dependencies.map((field) => JSON.stringify([identity, step.publicationGeneration, step.index, field])),
  };
}

/** Rebuild one frame catalog from its non-expression fields, then expose only
 * definitions whose same-frame dependencies are present. Removed definitions
 * therefore disappear cleanly instead of lingering in a prior augmented row. */
export function augmentStepWithExpressionDerivedFields(
  step: StepInfo,
  definitions: readonly ExpressionDerivedFieldDefinition[],
): StepInfo {
  const expressionIds = new Set(Object.entries(step.fieldOrigins ?? {})
    .filter(([, origin]) => origin === "user_expression")
    .map(([field]) => field));
  const baseFields = step.fields.filter((field) => !expressionIds.has(field));
  const baseUnits = Object.fromEntries(Object.entries(step.fieldUnits).filter(([field]) => !expressionIds.has(field)));
  const baseOrigins = Object.fromEntries(Object.entries(step.fieldOrigins ?? {}).filter(([, origin]) => origin !== "user_expression"));
  const available = definitions.filter(({ dependencies }) => dependencies.every((field) => baseFields.includes(field)));
  return {
    ...step,
    fields: [...baseFields, ...available.map(({ id }) => id)],
    fieldUnits: { ...baseUnits, ...Object.fromEntries(available.map(({ id, unit }) => [id, unit])) },
    fieldOrigins: { ...baseOrigins, ...Object.fromEntries(available.map(({ id }) => [id, "user_expression" as const])) },
  };
}
