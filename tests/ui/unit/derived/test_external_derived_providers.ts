import {
  EXTERNAL_DERIVED_PROVIDER_RESULT_SCHEMA,
  EXTERNAL_DERIVED_PROVIDER_SCHEMA,
  EXTERNAL_DERIVED_PROVIDER_STORAGE_KEY,
  MAX_EXTERNAL_PROVIDER_STORE_BYTES,
  clearExternalDerivedProviders,
  executeExternalDerivedProvider,
  externalDerivedFieldCacheIdentity,
  importExternalDerivedProvider,
  loadExternalDerivedProviders,
  parseExternalDerivedProvider,
  saveExternalDerivedProviders,
  setExternalDerivedProviderEnabled,
  type ExternalDerivedProvider,
  type ExternalDerivedProviderAdapter,
  type ExternalProviderExecutionRequest,
  type ExternalProviderWireResult,
} from "../../../../ui/src/lib/derived/externalDerivedProviders.ts";
import {
  assessExternalProviderForStep,
  augmentStepWithExternalProviders,
} from "../../../../ui/src/lib/derived/externalDerivedCatalog.ts";
import type { StepInfo } from "../../../../ui/src/lib/run/runOutputs.ts";
import { clearFieldCache, getOrLoadField, makeFieldKey } from "../../../../ui/src/lib/core/fieldCache.ts";

let passed = 0;
const failures: string[] = [];
const check = (condition: boolean, label: string) => condition ? passed += 1 : failures.push(label);
const throws = (fn: () => unknown, pattern: RegExp, label: string) => {
  try { fn(); failures.push(label); }
  catch (error) { check(pattern.test(error instanceof Error ? error.message : String(error)), label); }
};
const rejects = async (fn: () => Promise<unknown>, pattern: RegExp, label: string) => {
  try { await fn(); failures.push(label); }
  catch (error) { check(pattern.test(error instanceof Error ? `${error.name}: ${error.message}` : String(error)), label); }
};

class MemoryStorage {
  values = new Map<string, string>();
  getItem(key: string) { return this.values.get(key) ?? null; }
  setItem(key: string, value: string) { this.values.set(key, value); }
  removeItem(key: string) { this.values.delete(key); }
}

const provider: ExternalDerivedProvider = parseExternalDerivedProvider({
  schema: EXTERNAL_DERIVED_PROVIDER_SCHEMA,
  schemaVersion: 1,
  definitionVersion: 1,
  id: "wind_speed_python",
  label: "Python wind speed",
  description: "Computes horizontal wind speed from exact same-frame u and v fields.",
  providerVersion: "1.2.0",
  enabled: true,
  kind: "python",
  permissions: ["execute-python", "read-frame-fields"],
  capabilities: ["numpy", "derived-field", "deterministic"],
  inputs: [
    { fieldId: "u", alias: "u", unit: "m/s" },
    { fieldId: "v", alias: "v", unit: "m/s" },
  ],
  output: {
    fieldId: "python_wind_speed",
    label: "Python wind speed",
    unit: "m/s",
    category: "dynamics",
    shapePolicy: { kind: "match-input", inputAlias: "u" },
  },
  entrypoint: {
    runtime: "pyodide",
    runtimeVersion: "0.26.4",
    source: "result = np.sqrt(u*u + v*v)",
    sourceDigest: "sha256:7c413f8ad8c4d9cf1e255a69699791b93593aa84f654092b73999babc98eb274",
    resultVariable: "result",
  },
});

check(provider.permissions.join(",") === "execute-python,read-frame-fields", "manifest canonicalizes permission order");
check(provider.capabilities.includes("derived-field") && provider.output.shapePolicy.kind === "match-input", "manifest retains explicit capabilities and shape policy");

const identity = externalDerivedFieldCacheIdentity(provider, "dataset-a");
for (const changed of [
  { ...provider, inputs: provider.inputs.map(input => input.alias === "u" ? { ...input, fieldId: "w" } : input) },
  { ...provider, inputs: provider.inputs.map(input => ({ ...input, unit: "K" })) },
  { ...provider, output: { ...provider.output, unit: "K" } },
  { ...provider, output: { ...provider.output, shapePolicy: { kind: "exact" as const, dimensions: [3] } } },
  { ...provider, permissions: [...provider.permissions, "network-access" as const] },
  { ...provider, enabled: false },
]) check(externalDerivedFieldCacheIdentity(changed, "dataset-a") !== identity, "same-version contract redefinitions cannot inherit old field values or admission");
check(externalDerivedFieldCacheIdentity(provider, "dataset-b") !== identity, "cached provider provenance stays bound to its registered dataset");
clearFieldCache();
let cacheLoads = 0;
const changedInput = { ...provider, inputs: provider.inputs.map(input => input.alias === "u" ? { ...input, fieldId: "w" } : input) };
const read = (definition: ExternalDerivedProvider) => getOrLoadField(
  makeFieldKey("/same-run", 4, externalDerivedFieldCacheIdentity(definition, "dataset-a"), "generation-4"),
  async () => { cacheLoads++; return { shape: [2], data: Float32Array.of(definition.inputs[0].fieldId === "u" ? 5 : 13, 2) }; },
);
const initial = await read(provider);
check(await read(provider) === initial && cacheLoads === 1, "unchanged provider requests reuse their immutable computation");
check((await read(changedInput)).data[0] === 13 && cacheLoads === 2, "a changed same-version input mapping executes its new numerical source instead of returning the cached old one");
clearFieldCache();
const catalogStep: StepInfo = {
  index: 4,
  simulationTimeS: 240,
  fields: ["u", "v"],
  grid: { nr: 2, nth: 1, nz: 1 },
  fieldUnits: { u: "m/s", v: "m/s" },
  fieldOrigins: { u: "native", v: "native" },
  publicationState: "complete",
  publicationGeneration: "generation-4",
};
const readyRuntime = () => ({ available: true, label: "Pyodide 0.26.4", reason: null });
const augmentedCatalog = augmentStepWithExternalProviders(catalogStep, [provider], readyRuntime);
check(augmentedCatalog.fields.includes("python_wind_speed") && augmentedCatalog.fieldOrigins?.python_wind_speed === "external_provider", "enabled qualified provider output enters the normal field catalog");
check(assessExternalProviderForStep({ ...catalogStep, fieldUnits: { u: "kt", v: "m/s" } }, provider, readyRuntime()).status === "unit_mismatch", "catalog qualification rejects input-unit mismatches before execution");
check(assessExternalProviderForStep(catalogStep, { ...provider, enabled: false }, readyRuntime()).status === "disabled", "disabled providers stay outside the field catalog");
throws(() => parseExternalDerivedProvider({ ...provider, schemaVersion: 2 }), /unsupported .*schema or version/, "unknown manifest schema version fails closed");
throws(() => parseExternalDerivedProvider({ ...provider, surprise: true }), /unsupported properties/, "unknown manifest properties fail closed");
throws(() => parseExternalDerivedProvider({ ...provider, providerVersion: "latest" }), /exact semantic version/, "provider version tags are rejected");
throws(() => parseExternalDerivedProvider({ ...provider, providerVersion: "^1.2.0" }), /exact semantic version/, "provider version ranges are rejected");
throws(() => parseExternalDerivedProvider({ ...provider, permissions: ["read-frame-fields"] }), /execute-python/, "Python execution permission is mandatory");
throws(() => parseExternalDerivedProvider({ ...provider, capabilities: ["numpy"] }), /derived-field/, "derived-field capability is mandatory");
throws(() => parseExternalDerivedProvider({ ...provider, inputs: [...provider.inputs, provider.inputs[0]] }), /fieldIds must be unique/, "duplicate input fields are rejected");
throws(() => parseExternalDerivedProvider({ ...provider, output: { ...provider.output, shapePolicy: { kind: "match-input", inputAlias: "w" } } }), /undeclared input alias/, "shape policy cannot reference an undeclared input");
throws(() => parseExternalDerivedProvider({ ...provider, output: { ...provider.output, fieldId: "u" } }), /must not also be one of its inputs/, "provider outputs cannot create a recursive input cycle");
throws(() => parseExternalDerivedProvider({ ...provider, entrypoint: { ...provider.entrypoint, sourceDigest: "sha256:nope" } }), /sha256 digest/, "malformed source integrity pins are rejected");

const imported = importExternalDerivedProvider(JSON.stringify(provider));
check(!imported.enabled, "import always registers a provider disabled for review");
throws(() => importExternalDerivedProvider("not-json"), /not valid JSON/, "malformed imports fail closed");

const storage = new MemoryStorage();
saveExternalDerivedProviders(storage, [provider], "2026-07-19T12:00:00.000Z");
const loaded = loadExternalDerivedProviders(storage);
check(loaded.warnings.length === 0 && loaded.providers[0]?.id === provider.id, "versioned provider registry persists and restores");
check(storage.values.get(EXTERNAL_DERIVED_PROVIDER_STORAGE_KEY)?.includes("external-derived-provider-set") === true, "storage uses the versioned registry envelope");
check(!setExternalDerivedProviderEnabled([provider], provider.id, false)[0].enabled, "provider enablement changes through a validated registry update");
throws(() => setExternalDerivedProviderEnabled([provider], "missing", false), /unknown external provider/, "unknown provider enablement fails closed");
throws(() => saveExternalDerivedProviders(storage, [provider, provider]), /IDs must be unique/, "duplicate provider IDs cannot persist");
storage.values.set(EXTERNAL_DERIVED_PROVIDER_STORAGE_KEY, "x".repeat(MAX_EXTERNAL_PROVIDER_STORE_BYTES + 1));
check(loadExternalDerivedProviders(storage).providers.length === 0 && loadExternalDerivedProviders(storage).warnings.length === 1, "oversized stored registries are rejected before JSON parsing");
clearExternalDerivedProviders(storage);
check(storage.getItem(EXTERNAL_DERIVED_PROVIDER_STORAGE_KEY) === null, "provider registry can be explicitly cleared");

const wire = (request: ExternalProviderExecutionRequest): ExternalProviderWireResult => ({
  schema: EXTERNAL_DERIVED_PROVIDER_RESULT_SCHEMA,
  schemaVersion: 1,
  requestId: request.requestId,
  providerId: request.provider.id,
  providerVersion: request.provider.providerVersion,
  fieldId: request.provider.output.fieldId,
  unit: request.provider.output.unit,
  shape: [2],
  encoding: "float32-le",
  values: Float32Array.of(5, 13).buffer,
  diagnostics: [{ severity: "info", code: "qualified", message: "Two cells computed" }],
  executionMs: 3.5,
});
const adapter: ExternalDerivedProviderAdapter = {
  adapterId: "test-python",
  kind: "python",
  supports: () => ({ available: true, label: "test", reason: null }),
  execute: async (request) => wire(request),
};
const controller = new AbortController();
const options = {
  requestId: "request-1",
  datasetIdentity: "cm1:/case",
  frame: { index: 4, simulationTimeS: 240, publicationGeneration: "generation-4" },
  grid: null,
  inputs: {
    u: { field: { shape: [2], data: Float32Array.of(3, 5) }, unit: "m/s", identity: "u@4" },
    v: { field: { shape: [2], data: Float32Array.of(4, 12) }, unit: "m/s", identity: "v@4" },
  },
  grantedPermissions: new Set(["execute-python", "read-frame-fields"] as const),
  signal: controller.signal,
};
const result = await executeExternalDerivedProvider(provider, adapter, options);
check(result.field.data[0] === 5 && result.field.data[1] === 13 && result.field.shape[0] === 2, "adapter wire bytes become a validated Float32 field");
check(result.provenance.providerVersion === "1.2.0" && result.provenance.inputIdentities.u === "u@4", "execution provenance binds provider version and exact input identities");
check(result.diagnostics[0]?.code === "qualified" && result.provenance.executionMs === 3.5, "bounded runtime diagnostics and timing survive qualification");

await rejects(() => executeExternalDerivedProvider({ ...provider, enabled: false }, adapter, options), /disabled/, "disabled providers cannot execute");
await rejects(() => executeExternalDerivedProvider(provider, adapter, { ...options, grantedPermissions: new Set(["read-frame-fields"] as const) }), /permissions were not granted.*execute-python/, "declared permissions require separate execution grants");
await rejects(() => executeExternalDerivedProvider(provider, adapter, { ...options, inputs: { ...options.inputs, u: { ...options.inputs.u, unit: "kt" } } }), /requires unit 'm\/s'/, "input units are qualified before adapter execution");

const wrongShapeAdapter: ExternalDerivedProviderAdapter = { ...adapter, execute: async (request) => ({ ...wire(request), shape: [1], values: Float32Array.of(5).buffer }) };
await rejects(() => executeExternalDerivedProvider(provider, wrongShapeAdapter, options), /violates declared shape policy/, "adapter output shape must satisfy the manifest policy");
const wrongBufferAdapter: ExternalDerivedProviderAdapter = { ...adapter, execute: async (request) => ({ ...wire(request), values: Float32Array.of(5).buffer }) };
await rejects(() => executeExternalDerivedProvider(provider, wrongBufferAdapter, options), /exactly 8 bytes/, "adapter output byte length must exactly match its shape");
const unavailableAdapter: ExternalDerivedProviderAdapter = { ...adapter, supports: () => ({ available: false, label: "test", reason: "runtime is absent" }) };
await rejects(() => executeExternalDerivedProvider(provider, unavailableAdapter, options), /runtime is absent/, "unavailable runtimes fail before execution");

const preCancelled = new AbortController();
preCancelled.abort();
await rejects(() => executeExternalDerivedProvider(provider, adapter, { ...options, signal: preCancelled.signal }), /AbortError:.*cancelled/, "pre-cancelled execution never enters the adapter");
const duringController = new AbortController();
const cancellingAdapter: ExternalDerivedProviderAdapter = {
  ...adapter,
  execute: async (request) => { duringController.abort(); return wire(request); },
};
await rejects(() => executeExternalDerivedProvider(provider, cancellingAdapter, { ...options, signal: duringController.signal }), /AbortError:.*cancelled/, "cancellation is rechecked after the runtime returns");

console.log(`external derived providers: ${passed} checks passed, ${failures.length} failed`);
if (failures.length) {
  failures.forEach((failure) => console.error(`FAIL ${failure}`));
  process.exit(1);
}
