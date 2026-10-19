import type { GridInfo } from "@/lib/run/runOutputs";
import { validateFieldShape, type NpyN } from "@/lib/platform/npy";

export const EXTERNAL_DERIVED_PROVIDER_SCHEMA = "nimbus.studio.external-derived-provider" as const;
export const EXTERNAL_DERIVED_PROVIDER_SCHEMA_VERSION = 1 as const;
export const EXTERNAL_DERIVED_PROVIDER_SET_SCHEMA = "nimbus.studio.external-derived-provider-set" as const;
export const EXTERNAL_DERIVED_PROVIDER_REQUEST_SCHEMA = "nimbus.studio.external-derived-provider-request" as const;
export const EXTERNAL_DERIVED_PROVIDER_RESULT_SCHEMA = "nimbus.studio.external-derived-provider-result" as const;
export const EXTERNAL_DERIVED_PROVIDER_PROVENANCE_SCHEMA = "nimbus.studio.external-derived-provider-provenance" as const;

export const MAX_EXTERNAL_DERIVED_PROVIDERS = 12;
export const MAX_EXTERNAL_PROVIDER_INPUTS = 16;
export const MAX_EXTERNAL_PROVIDER_SOURCE_CHARS = 24_000;
export const MAX_EXTERNAL_PROVIDER_STORE_BYTES = 384 * 1024;
export const MAX_EXTERNAL_PROVIDER_DIAGNOSTICS = 32;
export const EXTERNAL_DERIVED_PROVIDER_STORAGE_KEY = "nimbus.studio.external-derived-providers.v1";

const IDENTIFIER = /^[a-z][a-z0-9_]{0,63}$/;
const EXACT_VERSION = /^\d+\.\d+\.\d+(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?$/;
const SHA256 = /^sha256:[a-f0-9]{64}$/;

export type ExternalDerivedProviderKind = "python" | "plugin";
export type ExternalDerivedProviderPermission =
  | "read-frame-fields"
  | "execute-python"
  | "execute-plugin"
  | "network-access"
  | "filesystem-read"
  | "filesystem-write";
export type ExternalDerivedProviderCapability =
  | "derived-field"
  | "numpy"
  | "deterministic"
  | "native-plugin"
  | "chunk-aware";
export type ExternalDerivedFieldCategory =
  | "dynamics"
  | "thermodynamics"
  | "microphysics"
  | "radar"
  | "surface"
  | "custom";

export interface ExternalDerivedProviderInput {
  fieldId: string;
  alias: string;
  unit: string | null;
}

export type ExternalDerivedProviderShapePolicy =
  | { kind: "match-input"; inputAlias: string }
  | { kind: "exact"; dimensions: readonly number[] };

export interface ExternalDerivedProviderOutput {
  fieldId: string;
  label: string;
  unit: string;
  category: ExternalDerivedFieldCategory;
  shapePolicy: ExternalDerivedProviderShapePolicy;
}

export interface PythonProviderEntrypoint {
  runtime: "pyodide";
  runtimeVersion: string;
  source: string;
  sourceDigest: string;
  resultVariable: "result";
}

export interface PluginProviderEntrypoint {
  runtime: "nimbus-plugin-host";
  protocolVersion: 1;
  hostVersion: string;
  pluginId: string;
  exportName: string;
  artifactDigest: string;
}

interface ExternalDerivedProviderBase {
  schema: typeof EXTERNAL_DERIVED_PROVIDER_SCHEMA;
  schemaVersion: typeof EXTERNAL_DERIVED_PROVIDER_SCHEMA_VERSION;
  definitionVersion: 1;
  id: string;
  label: string;
  description: string;
  providerVersion: string;
  enabled: boolean;
  permissions: readonly ExternalDerivedProviderPermission[];
  capabilities: readonly ExternalDerivedProviderCapability[];
  inputs: readonly ExternalDerivedProviderInput[];
  output: ExternalDerivedProviderOutput;
}

export interface PythonDerivedProvider extends ExternalDerivedProviderBase {
  kind: "python";
  entrypoint: PythonProviderEntrypoint;
}

export interface PluginDerivedProvider extends ExternalDerivedProviderBase {
  kind: "plugin";
  entrypoint: PluginProviderEntrypoint;
}

export type ExternalDerivedProvider = PythonDerivedProvider | PluginDerivedProvider;

export function externalDerivedFieldCacheIdentity(provider: ExternalDerivedProvider, datasetIdentity: string): string {
  return JSON.stringify(["external", datasetIdentity, provider]);
}

export interface ExternalDerivedProviderLoadResult {
  providers: ExternalDerivedProvider[];
  warnings: string[];
}

export interface ExternalProviderStorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export interface ExternalProviderInputField {
  field: NpyN;
  unit: string | null;
  identity: string;
}

export interface ExternalProviderFrameIdentity {
  index: number;
  simulationTimeS: number | null;
  publicationGeneration: string | null;
}

export interface ExternalProviderExecutionRequest {
  schema: typeof EXTERNAL_DERIVED_PROVIDER_REQUEST_SCHEMA;
  schemaVersion: 1;
  requestId: string;
  datasetIdentity: string;
  frame: ExternalProviderFrameIdentity;
  grid: GridInfo | null;
  provider: ExternalDerivedProvider;
  inputs: Readonly<Record<string, ExternalProviderInputField>>;
}

export interface ExternalProviderDiagnostic {
  severity: "info" | "warning" | "error";
  code: string;
  message: string;
}

/** Structured-clone-safe response returned by a runtime adapter. The buffer is
 * always little-endian Float32 and must contain exactly product(shape) values. */
export interface ExternalProviderWireResult {
  schema: typeof EXTERNAL_DERIVED_PROVIDER_RESULT_SCHEMA;
  schemaVersion: 1;
  requestId: string;
  providerId: string;
  providerVersion: string;
  fieldId: string;
  unit: string;
  shape: readonly number[];
  encoding: "float32-le";
  values: ArrayBuffer;
  diagnostics: readonly ExternalProviderDiagnostic[];
  executionMs: number;
}

export interface ExternalProviderRuntimeStatus {
  available: boolean;
  label: string;
  reason: string | null;
}

export interface ExternalDerivedProviderAdapter {
  adapterId: string;
  kind: ExternalDerivedProviderKind;
  supports(provider: ExternalDerivedProvider): ExternalProviderRuntimeStatus;
  execute(request: ExternalProviderExecutionRequest, signal: AbortSignal): Promise<ExternalProviderWireResult>;
}

export interface ExternalProviderProvenance {
  schema: typeof EXTERNAL_DERIVED_PROVIDER_PROVENANCE_SCHEMA;
  schemaVersion: 1;
  requestId: string;
  datasetIdentity: string;
  frame: ExternalProviderFrameIdentity;
  providerId: string;
  providerVersion: string;
  providerKind: ExternalDerivedProviderKind;
  runtime: string;
  runtimeVersion: string;
  outputFieldId: string;
  outputUnit: string;
  inputIdentities: Readonly<Record<string, string>>;
  grantedPermissions: readonly ExternalDerivedProviderPermission[];
  executionMs: number;
  diagnostics: readonly ExternalProviderDiagnostic[];
}

export interface ExternalProviderExecutionResult {
  field: NpyN;
  diagnostics: readonly ExternalProviderDiagnostic[];
  provenance: ExternalProviderProvenance;
}

const PROVIDER_KEYS = new Set([
  "schema", "schemaVersion", "definitionVersion", "id", "label", "description",
  "providerVersion", "enabled", "kind", "permissions", "capabilities", "inputs",
  "output", "entrypoint",
]);
const INPUT_KEYS = new Set(["fieldId", "alias", "unit"]);
const OUTPUT_KEYS = new Set(["fieldId", "label", "unit", "category", "shapePolicy"]);
const PYTHON_ENTRYPOINT_KEYS = new Set(["runtime", "runtimeVersion", "source", "sourceDigest", "resultVariable"]);
const PLUGIN_ENTRYPOINT_KEYS = new Set(["runtime", "protocolVersion", "hostVersion", "pluginId", "exportName", "artifactDigest"]);
const MATCH_SHAPE_KEYS = new Set(["kind", "inputAlias"]);
const EXACT_SHAPE_KEYS = new Set(["kind", "dimensions"]);
const SET_KEYS = new Set(["schema", "schemaVersion", "savedAt", "providers"]);
const WIRE_RESULT_KEYS = new Set([
  "schema", "schemaVersion", "requestId", "providerId", "providerVersion", "fieldId",
  "unit", "shape", "encoding", "values", "diagnostics", "executionMs",
]);
const DIAGNOSTIC_KEYS = new Set(["severity", "code", "message"]);
const PERMISSIONS = new Set<ExternalDerivedProviderPermission>([
  "read-frame-fields", "execute-python", "execute-plugin", "network-access",
  "filesystem-read", "filesystem-write",
]);
const CAPABILITIES = new Set<ExternalDerivedProviderCapability>([
  "derived-field", "numpy", "deterministic", "native-plugin", "chunk-aware",
]);
const CATEGORIES = new Set<ExternalDerivedFieldCategory>([
  "dynamics", "thermodynamics", "microphysics", "radar", "surface", "custom",
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function exactKeys(value: Record<string, unknown>, allowed: ReadonlySet<string>, label: string): void {
  const extras = Object.keys(value).filter((key) => !allowed.has(key));
  if (extras.length) throw new Error(`${label} contains unsupported properties: ${extras.join(", ")}`);
}

function text(value: unknown, label: string, max: number): string {
  if (typeof value !== "string") throw new Error(`${label} must be a string`);
  const normalized = value.trim();
  if (!normalized) throw new Error(`${label} must not be empty`);
  if (normalized.length > max) throw new Error(`${label} exceeds ${max} characters`);
  return normalized;
}

function identifier(value: unknown, label: string): string {
  const id = text(value, label, 64);
  if (!IDENTIFIER.test(id)) throw new Error(`${label} '${id}' is not a portable lowercase field identifier`);
  return id;
}

function exactVersion(value: unknown, label: string): string {
  const version = text(value, label, 80);
  if (!EXACT_VERSION.test(version)) throw new Error(`${label} must be an exact semantic version, not a range or tag`);
  return version;
}

function digest(value: unknown, label: string): string {
  const normalized = text(value, label, 71);
  if (!SHA256.test(normalized)) throw new Error(`${label} must be a lowercase sha256 digest`);
  return normalized;
}

function stringEnum<T extends string>(value: unknown, allowed: ReadonlySet<T>, label: string): T {
  if (typeof value !== "string" || !allowed.has(value as T)) throw new Error(`${label} is unsupported`);
  return value as T;
}

function uniqueEnumList<T extends string>(value: unknown, allowed: ReadonlySet<T>, label: string): T[] {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array`);
  const parsed = value.map((entry) => stringEnum(entry, allowed, label));
  if (new Set(parsed).size !== parsed.length) throw new Error(`${label} must not contain duplicates`);
  return parsed.sort();
}

function parseInput(value: unknown, index: number): ExternalDerivedProviderInput {
  if (!isRecord(value)) throw new Error(`input ${index + 1} must be an object`);
  exactKeys(value, INPUT_KEYS, `input ${index + 1}`);
  const unit = value.unit === null ? null : text(value.unit, `input ${index + 1} unit`, 48);
  return {
    fieldId: identifier(value.fieldId, `input ${index + 1} fieldId`),
    alias: identifier(value.alias, `input ${index + 1} alias`),
    unit,
  };
}

function parseShapePolicy(value: unknown, aliases: ReadonlySet<string>): ExternalDerivedProviderShapePolicy {
  if (!isRecord(value)) throw new Error("output shapePolicy must be an object");
  if (value.kind === "match-input") {
    exactKeys(value, MATCH_SHAPE_KEYS, "match-input shape policy");
    const inputAlias = identifier(value.inputAlias, "shape policy inputAlias");
    if (!aliases.has(inputAlias)) throw new Error(`shape policy references undeclared input alias '${inputAlias}'`);
    return { kind: "match-input", inputAlias };
  }
  if (value.kind === "exact") {
    exactKeys(value, EXACT_SHAPE_KEYS, "exact shape policy");
    if (!Array.isArray(value.dimensions)) throw new Error("exact shape dimensions must be an array");
    validateFieldShape(value.dimensions as number[]);
    return { kind: "exact", dimensions: [...value.dimensions as number[]] };
  }
  throw new Error("output shapePolicy kind is unsupported");
}

function parseOutput(value: unknown, aliases: ReadonlySet<string>): ExternalDerivedProviderOutput {
  if (!isRecord(value)) throw new Error("output must be an object");
  exactKeys(value, OUTPUT_KEYS, "output");
  return {
    fieldId: identifier(value.fieldId, "output fieldId"),
    label: text(value.label, "output label", 120),
    unit: text(value.unit, "output unit", 48),
    category: stringEnum(value.category, CATEGORIES, "output category"),
    shapePolicy: parseShapePolicy(value.shapePolicy, aliases),
  };
}

function parseEntrypoint(kind: ExternalDerivedProviderKind, value: unknown): PythonProviderEntrypoint | PluginProviderEntrypoint {
  if (!isRecord(value)) throw new Error("entrypoint must be an object");
  if (kind === "python") {
    exactKeys(value, PYTHON_ENTRYPOINT_KEYS, "Python entrypoint");
    if (value.runtime !== "pyodide" || value.resultVariable !== "result") throw new Error("unsupported Python entrypoint contract");
    return {
      runtime: "pyodide",
      runtimeVersion: exactVersion(value.runtimeVersion, "Pyodide runtimeVersion"),
      source: text(value.source, "Python source", MAX_EXTERNAL_PROVIDER_SOURCE_CHARS),
      sourceDigest: digest(value.sourceDigest, "Python sourceDigest"),
      resultVariable: "result",
    };
  }
  exactKeys(value, PLUGIN_ENTRYPOINT_KEYS, "plugin entrypoint");
  if (value.runtime !== "nimbus-plugin-host" || value.protocolVersion !== 1) throw new Error("unsupported plugin entrypoint contract");
  return {
    runtime: "nimbus-plugin-host",
    protocolVersion: 1,
    hostVersion: exactVersion(value.hostVersion, "plugin hostVersion"),
    pluginId: identifier(value.pluginId, "pluginId"),
    exportName: identifier(value.exportName, "plugin exportName"),
    artifactDigest: digest(value.artifactDigest, "plugin artifactDigest"),
  };
}

export function parseExternalDerivedProvider(value: unknown): ExternalDerivedProvider {
  if (!isRecord(value)) throw new Error("external provider must be an object");
  exactKeys(value, PROVIDER_KEYS, "external provider");
  if (value.schema !== EXTERNAL_DERIVED_PROVIDER_SCHEMA || value.schemaVersion !== 1 || value.definitionVersion !== 1) {
    throw new Error("unsupported external provider schema or version");
  }
  if (typeof value.enabled !== "boolean") throw new Error("provider enabled must be boolean");
  const kind = value.kind === "python" || value.kind === "plugin" ? value.kind : null;
  if (!kind) throw new Error("provider kind must be python or plugin");
  if (!Array.isArray(value.inputs) || value.inputs.length === 0 || value.inputs.length > MAX_EXTERNAL_PROVIDER_INPUTS) {
    throw new Error(`provider inputs must contain 1-${MAX_EXTERNAL_PROVIDER_INPUTS} fields`);
  }
  const inputs = value.inputs.map(parseInput);
  if (new Set(inputs.map(({ fieldId }) => fieldId)).size !== inputs.length) throw new Error("provider input fieldIds must be unique");
  if (new Set(inputs.map(({ alias }) => alias)).size !== inputs.length) throw new Error("provider input aliases must be unique");
  const permissions = uniqueEnumList(value.permissions, PERMISSIONS, "provider permissions");
  const capabilities = uniqueEnumList(value.capabilities, CAPABILITIES, "provider capabilities");
  if (!capabilities.includes("derived-field")) throw new Error("provider must declare the derived-field capability");
  if (!permissions.includes("read-frame-fields")) throw new Error("provider must declare read-frame-fields permission");
  if (kind === "python" && (!permissions.includes("execute-python") || !capabilities.includes("numpy"))) {
    throw new Error("Python providers require execute-python permission and numpy capability");
  }
  if (kind === "plugin" && (!permissions.includes("execute-plugin") || !capabilities.includes("native-plugin"))) {
    throw new Error("plugin providers require execute-plugin permission and native-plugin capability");
  }
  const output = parseOutput(value.output, new Set(inputs.map(({ alias }) => alias)));
  if (inputs.some(({ fieldId }) => fieldId === output.fieldId)) {
    throw new Error("provider output fieldId must not also be one of its inputs");
  }
  const base = {
    schema: EXTERNAL_DERIVED_PROVIDER_SCHEMA,
    schemaVersion: EXTERNAL_DERIVED_PROVIDER_SCHEMA_VERSION,
    definitionVersion: 1 as const,
    id: identifier(value.id, "provider id"),
    label: text(value.label, "provider label", 120),
    description: text(value.description, "provider description", 600),
    providerVersion: exactVersion(value.providerVersion, "providerVersion"),
    enabled: value.enabled,
    permissions,
    capabilities,
    inputs,
    output,
  };
  const entrypoint = parseEntrypoint(kind, value.entrypoint);
  return kind === "python"
    ? { ...base, kind, entrypoint: entrypoint as PythonProviderEntrypoint }
    : { ...base, kind, entrypoint: entrypoint as PluginProviderEntrypoint };
}

/** Import is deliberately a review step: even a manifest that says enabled is
 * registered disabled until the user explicitly enables it in Studio. */
export function importExternalDerivedProvider(serialized: string): ExternalDerivedProvider {
  if (serialized.length > MAX_EXTERNAL_PROVIDER_STORE_BYTES) throw new Error("provider manifest exceeds the import byte limit");
  let value: unknown;
  try { value = JSON.parse(serialized); }
  catch { throw new Error("provider manifest is not valid JSON"); }
  return { ...parseExternalDerivedProvider(value), enabled: false } as ExternalDerivedProvider;
}

function canonicalProviders(providers: readonly ExternalDerivedProvider[]): ExternalDerivedProvider[] {
  if (providers.length > MAX_EXTERNAL_DERIVED_PROVIDERS) throw new Error(`at most ${MAX_EXTERNAL_DERIVED_PROVIDERS} external providers may be registered`);
  const parsed = providers.map(parseExternalDerivedProvider);
  if (new Set(parsed.map(({ id }) => id)).size !== parsed.length) throw new Error("provider IDs must be unique");
  if (new Set(parsed.map(({ output }) => output.fieldId)).size !== parsed.length) throw new Error("external provider output field IDs must be unique");
  return parsed.sort((left, right) => left.label.localeCompare(right.label));
}

export function saveExternalDerivedProviders(
  storage: ExternalProviderStorageLike,
  providers: readonly ExternalDerivedProvider[],
  savedAt = new Date().toISOString(),
): void {
  if (!Number.isFinite(Date.parse(savedAt))) throw new Error("savedAt must be an ISO timestamp");
  const payload = JSON.stringify({
    schema: EXTERNAL_DERIVED_PROVIDER_SET_SCHEMA,
    schemaVersion: 1,
    savedAt,
    providers: canonicalProviders(providers),
  });
  if (new TextEncoder().encode(payload).byteLength > MAX_EXTERNAL_PROVIDER_STORE_BYTES) {
    throw new Error(`provider registry exceeds the ${MAX_EXTERNAL_PROVIDER_STORE_BYTES}-byte persistence limit`);
  }
  storage.setItem(EXTERNAL_DERIVED_PROVIDER_STORAGE_KEY, payload);
}

export function loadExternalDerivedProviders(storage: ExternalProviderStorageLike): ExternalDerivedProviderLoadResult {
  let raw: string | null;
  try { raw = storage.getItem(EXTERNAL_DERIVED_PROVIDER_STORAGE_KEY); }
  catch (error) { return { providers: [], warnings: [`Provider storage is unavailable: ${error instanceof Error ? error.message : String(error)}`] }; }
  if (raw == null) return { providers: [], warnings: [] };
  if (new TextEncoder().encode(raw).byteLength > MAX_EXTERNAL_PROVIDER_STORE_BYTES) {
    return { providers: [], warnings: ["Stored provider registry was rejected because it exceeds the persistence limit"] };
  }
  try {
    const value: unknown = JSON.parse(raw);
    if (!isRecord(value)) throw new Error("stored provider registry must be an object");
    exactKeys(value, SET_KEYS, "stored provider registry");
    if (value.schema !== EXTERNAL_DERIVED_PROVIDER_SET_SCHEMA || value.schemaVersion !== 1 || typeof value.savedAt !== "string" || !Number.isFinite(Date.parse(value.savedAt)) || !Array.isArray(value.providers)) {
      throw new Error("stored provider registry has an invalid schema, version, or timestamp");
    }
    return { providers: canonicalProviders(value.providers as ExternalDerivedProvider[]), warnings: [] };
  } catch (error) {
    return { providers: [], warnings: [`Stored providers were rejected: ${error instanceof Error ? error.message : String(error)}`] };
  }
}

export function clearExternalDerivedProviders(storage: ExternalProviderStorageLike): void {
  storage.removeItem(EXTERNAL_DERIVED_PROVIDER_STORAGE_KEY);
}

export function setExternalDerivedProviderEnabled(
  providers: readonly ExternalDerivedProvider[],
  providerId: string,
  enabled: boolean,
): ExternalDerivedProvider[] {
  if (!providers.some(({ id }) => id === providerId)) throw new Error(`unknown external provider '${providerId}'`);
  return canonicalProviders(providers.map((provider) => provider.id === providerId ? { ...provider, enabled } as ExternalDerivedProvider : provider));
}

function abortError(): Error {
  const error = new Error("External provider execution was cancelled");
  error.name = "AbortError";
  return error;
}

function assertNotAborted(signal: AbortSignal): void {
  if (signal.aborted) throw abortError();
}

function expectedOutputShape(provider: ExternalDerivedProvider, inputs: Readonly<Record<string, ExternalProviderInputField>>): readonly number[] {
  const policy = provider.output.shapePolicy;
  if (policy.kind === "exact") return policy.dimensions;
  const match = inputs[policy.inputAlias];
  if (!match) throw new Error(`provider is missing shape-reference input '${policy.inputAlias}'`);
  return match.field.shape;
}

function validateRequestInputs(provider: ExternalDerivedProvider, inputs: Readonly<Record<string, ExternalProviderInputField>>): void {
  const expectedAliases = new Set(provider.inputs.map(({ alias }) => alias));
  const extras = Object.keys(inputs).filter((alias) => !expectedAliases.has(alias));
  if (extras.length) throw new Error(`provider request contains undeclared inputs: ${extras.join(", ")}`);
  for (const declaration of provider.inputs) {
    const input = inputs[declaration.alias];
    if (!input) throw new Error(`provider request is missing input '${declaration.alias}'`);
    const { elementCount } = validateFieldShape(input.field.shape);
    if (input.field.data.length !== elementCount) throw new Error(`provider input '${declaration.alias}' data length does not match its shape`);
    if (declaration.unit !== null && input.unit !== declaration.unit) {
      throw new Error(`provider input '${declaration.alias}' requires unit '${declaration.unit}', received '${input.unit ?? "unknown"}'`);
    }
    if (!input.identity) throw new Error(`provider input '${declaration.alias}' is missing provenance identity`);
  }
}

function parseDiagnostic(value: unknown, index: number): ExternalProviderDiagnostic {
  if (!isRecord(value)) throw new Error(`diagnostic ${index + 1} must be an object`);
  exactKeys(value, DIAGNOSTIC_KEYS, `diagnostic ${index + 1}`);
  return {
    severity: stringEnum(value.severity, new Set(["info", "warning", "error"]), `diagnostic ${index + 1} severity`),
    code: identifier(value.code, `diagnostic ${index + 1} code`),
    message: text(value.message, `diagnostic ${index + 1} message`, 1_000),
  };
}

function parseWireResult(value: unknown, request: ExternalProviderExecutionRequest): ExternalProviderWireResult {
  if (!isRecord(value)) throw new Error("external provider result must be an object");
  exactKeys(value, WIRE_RESULT_KEYS, "external provider result");
  if (value.schema !== EXTERNAL_DERIVED_PROVIDER_RESULT_SCHEMA || value.schemaVersion !== 1 || value.encoding !== "float32-le") throw new Error("unsupported external provider result contract");
  if (value.requestId !== request.requestId || value.providerId !== request.provider.id || value.providerVersion !== request.provider.providerVersion) throw new Error("external provider result identity does not match its request");
  if (value.fieldId !== request.provider.output.fieldId || value.unit !== request.provider.output.unit) throw new Error("external provider result output metadata does not match its manifest");
  if (!Array.isArray(value.shape)) throw new Error("external provider result shape must be an array");
  const shape = value.shape as number[];
  const { decodedBytes } = validateFieldShape(shape);
  const expectedShape = expectedOutputShape(request.provider, request.inputs);
  if (shape.length !== expectedShape.length || shape.some((dimension, axis) => dimension !== expectedShape[axis])) throw new Error(`external provider result shape ${shape.join("×")} violates declared shape policy ${expectedShape.join("×")}`);
  if (!(value.values instanceof ArrayBuffer) || value.values.byteLength !== decodedBytes) throw new Error(`external provider result buffer must contain exactly ${decodedBytes} bytes`);
  if (!Array.isArray(value.diagnostics) || value.diagnostics.length > MAX_EXTERNAL_PROVIDER_DIAGNOSTICS) throw new Error(`external provider result diagnostics exceed the ${MAX_EXTERNAL_PROVIDER_DIAGNOSTICS}-item limit`);
  if (typeof value.executionMs !== "number" || !Number.isFinite(value.executionMs) || value.executionMs < 0) throw new Error("external provider executionMs must be a finite non-negative number");
  return {
    schema: EXTERNAL_DERIVED_PROVIDER_RESULT_SCHEMA,
    schemaVersion: 1,
    requestId: value.requestId as string,
    providerId: value.providerId as string,
    providerVersion: value.providerVersion as string,
    fieldId: value.fieldId as string,
    unit: value.unit as string,
    shape: [...shape],
    encoding: "float32-le",
    values: value.values,
    diagnostics: value.diagnostics.map(parseDiagnostic),
    executionMs: value.executionMs,
  };
}

export async function executeExternalDerivedProvider(
  providerValue: ExternalDerivedProvider,
  adapter: ExternalDerivedProviderAdapter,
  options: {
    requestId: string;
    datasetIdentity: string;
    frame: ExternalProviderFrameIdentity;
    grid: GridInfo | null;
    inputs: Readonly<Record<string, ExternalProviderInputField>>;
    grantedPermissions: ReadonlySet<ExternalDerivedProviderPermission>;
    signal: AbortSignal;
  },
): Promise<ExternalProviderExecutionResult> {
  const provider = parseExternalDerivedProvider(providerValue);
  if (!provider.enabled) throw new Error(`external provider '${provider.id}' is disabled`);
  if (adapter.kind !== provider.kind) throw new Error(`adapter '${adapter.adapterId}' cannot execute ${provider.kind} providers`);
  const missingPermissions = provider.permissions.filter((permission) => !options.grantedPermissions.has(permission));
  if (missingPermissions.length) throw new Error(`external provider permissions were not granted: ${missingPermissions.join(", ")}`);
  const status = adapter.supports(provider);
  if (!status.available) throw new Error(status.reason ?? `runtime '${status.label}' is unavailable`);
  text(options.requestId, "requestId", 160);
  text(options.datasetIdentity, "datasetIdentity", 1_000);
  validateRequestInputs(provider, options.inputs);
  assertNotAborted(options.signal);
  const request: ExternalProviderExecutionRequest = {
    schema: EXTERNAL_DERIVED_PROVIDER_REQUEST_SCHEMA,
    schemaVersion: 1,
    requestId: options.requestId,
    datasetIdentity: options.datasetIdentity,
    frame: options.frame,
    grid: options.grid,
    provider,
    inputs: options.inputs,
  };
  const result = parseWireResult(await adapter.execute(request, options.signal), request);
  assertNotAborted(options.signal);
  const runtime = provider.entrypoint.runtime;
  const runtimeVersion = provider.kind === "python" ? provider.entrypoint.runtimeVersion : provider.entrypoint.hostVersion;
  return {
    field: { shape: [...result.shape], data: new Float32Array(result.values) },
    diagnostics: result.diagnostics,
    provenance: {
      schema: EXTERNAL_DERIVED_PROVIDER_PROVENANCE_SCHEMA,
      schemaVersion: 1,
      requestId: request.requestId,
      datasetIdentity: request.datasetIdentity,
      frame: request.frame,
      providerId: provider.id,
      providerVersion: provider.providerVersion,
      providerKind: provider.kind,
      runtime,
      runtimeVersion,
      outputFieldId: provider.output.fieldId,
      outputUnit: provider.output.unit,
      inputIdentities: Object.fromEntries(provider.inputs.map(({ alias }) => [alias, request.inputs[alias].identity])),
      grantedPermissions: [...options.grantedPermissions].sort(),
      executionMs: result.executionMs,
      diagnostics: result.diagnostics,
    },
  };
}
