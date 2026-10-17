import { createLocalDatasetSource, datasetSourceIdentity } from "../../../../ui/src/lib/dataset/datasetSources.ts";
import {
  EXPRESSION_DERIVED_FIELD_PROVENANCE_SCHEMA,
  EXPRESSION_DERIVED_FIELD_SCHEMA,
  MAX_EXPRESSION_DERIVED_FIELDS,
  buildExpressionDerivedFieldProvenance,
  clearExpressionDerivedFields,
  compileExpressionDerivedField,
  createExpressionDerivedField,
  evaluateExpressionDerivedField,
  expressionDerivedFieldCacheIdentity,
  expressionDerivedFieldDefinitionIdentity,
  expressionDerivedFieldStorageKey,
  loadExpressionDerivedFields,
  parseExpressionDerivedField,
  saveExpressionDerivedFields,
  type StorageLike,
} from "../../../../ui/src/lib/derived/expressionDerivedFields.ts";
import type { StepInfo } from "../../../../ui/src/lib/run/runOutputs.ts";
import {
  evaluateExpressionDerivedInWorker,
  isExpressionEvaluationAbort,
  type ExpressionDerivedWorker,
} from "../../../../ui/src/lib/derived/expressionDerivedWorkerClient.ts";
import {
  evaluateExpressionWorkerRequest,
  type ExpressionDerivedWorkerReply,
  type ExpressionDerivedWorkerRequest,
} from "../../../../ui/src/lib/derived/expressionDerivedWorkerProtocol.ts";

let passed = 0;
const failures: string[] = [];
const check = (condition: unknown, label: string) => condition ? passed += 1 : failures.push(label);
const throws = (fn: () => unknown, pattern: RegExp, label: string) => {
  try { fn(); failures.push(label); }
  catch (error) { check(pattern.test(error instanceof Error ? error.message : String(error)), label); }
};

class MemoryStorage implements StorageLike {
  readonly values = new Map<string, string>();
  getItem(key: string) { return this.values.get(key) ?? null; }
  setItem(key: string, value: string) { this.values.set(key, value); }
  removeItem(key: string) { this.values.delete(key); }
}

const fields = new Set(["u", "v", "theta"]);
const wind = createExpressionDerivedField({
  id: "custom_wind",
  label: "Custom wind",
  unit: "m/s",
  dependencies: ["v", "u"],
  parameters: [{ id: "scale", value: 1 }],
  formula: "scale * sqrt(u^2 + v^2)",
}, { nativeFields: fields, availableFields: fields });

check(wind.schema === EXPRESSION_DERIVED_FIELD_SCHEMA && wind.definitionVersion === 1 && wind.provider === "nimbus_studio_user_expression", "versioned serializable definition contract");
check(Object.isFrozen(wind) && Object.isFrozen(wind.dependencies) && Object.isFrozen(wind.parameters[0]), "canonical definition is deeply immutable at contract leaves");
check(wind.dependencies.join(",") === "u,v" && wind.parameters[0].id === "scale", "dependencies and parameters receive stable canonical ordering");
check(parseExpressionDerivedField(JSON.parse(JSON.stringify(wind))).id === "custom_wind", "serialized definition round trip validates");
throws(() => parseExpressionDerivedField({ ...wind, futureSwitch: true }), /unsupported property/, "unknown definition properties fail closed");
throws(() => parseExpressionDerivedField({ ...wind, schemaVersion: 2 }), /unsupported .*schema or version/, "unknown schema versions fail closed");
throws(() => parseExpressionDerivedField({ ...wind, invalidValuePolicy: "fill zero" }), /invalid-value policy/, "invalid-value policy cannot silently change");

throws(() => createExpressionDerivedField({ ...wind, id: "Native-Wind" }), /must start with a lowercase/, "field IDs use strict portable identifiers");
throws(() => createExpressionDerivedField({ ...wind, id: "sqrt" }), /reserved/, "evaluator names are reserved");
throws(() => createExpressionDerivedField({ ...wind, id: "u" }, { nativeFields: fields }), /native dataset field/, "native fields take precedence over user definitions");
throws(() => createExpressionDerivedField({ ...wind, id: "temperature" }, { reservedFieldIds: new Set(["temperature"]) }), /existing dataset or registry field/, "built-in registry fields cannot be shadowed");
throws(() => createExpressionDerivedField({ ...wind, dependencies: ["u", "missing"] }, { availableFields: fields }), /not available.*missing/, "declared dependencies qualify against the dataset catalog");
throws(() => createExpressionDerivedField({ ...wind, dependencies: ["u", "u"] }), /duplicates/, "duplicate dependencies fail closed");
throws(() => createExpressionDerivedField({ ...wind, parameters: [{ id: "u", value: 2 }] }), /both a dependency and a parameter/, "dependency and parameter namespaces cannot overlap");
throws(() => createExpressionDerivedField({ ...wind, formula: "u@step:-1 + v" }), /cross-frame/, "registry expressions reject cross-frame references");
throws(() => createExpressionDerivedField({ ...wind, formula: "u + undeclared" }), /wasn't listed/, "formula identifiers must be declared");
throws(() => createExpressionDerivedField({ ...wind, formula: "u + v" }), /declared input 'scale'.*isn't referenced/, "unused parameters fail closed");
throws(() => createExpressionDerivedField({ ...wind, parameters: [{ id: "scale", value: Number.NaN }] }), /finite numeric/, "parameters must be finite numbers");

const compiled = compileExpressionDerivedField(wind, { nativeFields: fields, availableFields: fields });
const evaluated = evaluateExpressionDerivedField(compiled, {
  u: { shape: [2, 2], data: Float32Array.from([3, 5, Number.NaN, 8]) },
  v: { shape: [2, 2], data: Float32Array.from([4, 12, 1, 15]) },
});
check(evaluated.shape.join("×") === "2×2" && evaluated.data[0] === 5 && evaluated.data[1] === 13 && evaluated.data[3] === 17, "sandboxed expression evaluates exact per-cell values");
check(Number.isNaN(evaluated.data[2]), "nonfinite dependency propagates as NaN before formula evaluation");
check(
  evaluateExpressionWorkerRequest({
    definition: wind,
    inputs: {
      u: { shape: [1], data: Float32Array.of(3) },
      v: { shape: [1], data: Float32Array.of(4) },
    },
  }).data[0] === 5,
  "worker protocol recompiles the serializable definition and preserves numerical output",
);

class FakeExpressionWorker implements ExpressionDerivedWorker {
  onmessage: ((event: { data: ExpressionDerivedWorkerReply }) => void) | null = null;
  onerror: ((event: { message: string }) => void) | null = null;
  posted: ExpressionDerivedWorkerRequest | null = null;
  terminated = 0;
  postMessage(message: ExpressionDerivedWorkerRequest): void { this.posted = message; }
  terminate(): void { this.terminated += 1; }
  succeed(result: ReturnType<typeof evaluateExpressionWorkerRequest>): void {
    this.onmessage?.({ data: { ok: true, result } });
  }
}

{
  const backing = Float32Array.of(99, 3, 88);
  const source = backing.subarray(1, 2);
  const geometry = { sourceShape: [3], origin: [1], extent: [1], stride: [1],
    weighting: "uniform_cell_volume" as const, sampleLocation: "source_cell_center" as const,
    centroidsM: null };
  const worker = new FakeExpressionWorker();
  const pending = evaluateExpressionDerivedInWorker(wind, {
    u: { shape: [1], data: source, windowGeometry: geometry },
    v: { shape: [1], data: Float32Array.of(4) },
  }, { workerFactory: () => worker });
  check(worker.posted!.inputs.u.windowGeometry === undefined, "unused geometry does not cross the expression worker boundary");
  check(worker.posted!.inputs.u.data.buffer.byteLength === 4 && worker.posted!.inputs.u.data[0] === 3,
    "a small input view cannot clone its unrelated entire backing allocation");
  check(backing.byteLength === 12 && backing[0] === 99 && backing[2] === 88, "cached input storage remains attached and untouched");
  worker.succeed(evaluateExpressionWorkerRequest(worker.posted!));
  check((await pending).data[0] === 5, "visible-span copying preserves the same expression values");
}

{
  const worker = new FakeExpressionWorker();
  const pending = evaluateExpressionDerivedInWorker(wind, {
    u: { shape: [1], data: Float32Array.of(3) }, v: { shape: [1], data: Float32Array.of(4) },
  }, { workerFactory: () => worker, timeoutMs: 5 });
  let error: unknown;
  try { await pending; } catch (caught) { error = caught; }
  check(error instanceof Error && /startup or evaluation.*timeout/.test(error.message), "silent startup or execution fails within its bounded timeout");
  check(worker.terminated === 1, "a timed-out expression worker is retired");
}

{
  const worker = new FakeExpressionWorker();
  worker.postMessage = () => { throw new DOMException("clone failed", "DataCloneError"); };
  const pending = evaluateExpressionDerivedInWorker(wind, {
    u: { shape: [1], data: Float32Array.of(3) }, v: { shape: [1], data: Float32Array.of(4) },
  }, { workerFactory: () => worker });
  let error: unknown;
  try { await pending; } catch (caught) { error = caught; }
  check(error instanceof DOMException && error.name === "DataCloneError", "postMessage failures reject instead of running a UI-thread fallback");
  check(worker.terminated === 1, "postMessage failures retire their worker and timeout");
}

{
  const urls: string[] = [];
  const replacements: FakeExpressionWorker[] = [];
  class DefaultExpressionWorker extends FakeExpressionWorker {
    constructor(url: string) { super(); urls.push(url); replacements.push(this); }
  }
  Object.assign(globalThis, { Worker: DefaultExpressionWorker });
  for (let attempt = 0; attempt < 2; attempt++) {
    const pending = evaluateExpressionDerivedInWorker(wind, {
      u: { shape: [1], data: Float32Array.of(3) }, v: { shape: [1], data: Float32Array.of(4) },
    });
    const worker = replacements.at(-1)!;
    if (attempt === 0) worker.onerror?.({ message: "module startup blocked" });
    else worker.succeed(evaluateExpressionWorkerRequest(worker.posted!));
    try { const result = await pending; check(result.data[0] === 5, "a fresh replacement URL can complete after startup failure"); }
    catch (error) { check(error instanceof Error && error.message === "module startup blocked", "startup failures are surfaced without a CPU fallback"); }
    check(worker.terminated === 1, "each failed or successful worker retires exactly once");
  }
  check(urls.length === 2 && urls[0] !== urls[1] && urls.every(url => /session=.*&generation=\d+/.test(url)),
    "replacement expression workers use distinct URLs under the unchanged isolation policy");
  delete (globalThis as Record<string, unknown>).Worker;
}

{
  const worker = new FakeExpressionWorker();
  const controller = new AbortController();
  const pending = evaluateExpressionDerivedInWorker(wind, {
    u: { shape: [1], data: Float32Array.of(3) },
    v: { shape: [1], data: Float32Array.of(4) },
  }, { signal: controller.signal, workerFactory: () => worker });
  controller.abort();
  let error: unknown;
  try { await pending; } catch (caught) { error = caught; }
  check(isExpressionEvaluationAbort(error), "aborting an expression evaluation rejects with AbortError");
  check(worker.terminated === 1, "aborting actively terminates the expression worker");
  check(worker.posted?.definition.id === wind.id, "worker receives a serializable versioned definition");
}

{
  const worker = new FakeExpressionWorker();
  const pending = evaluateExpressionDerivedInWorker(wind, {
    u: { shape: [1], data: Float32Array.of(3) },
    v: { shape: [1], data: Float32Array.of(4) },
  }, { workerFactory: () => worker });
  worker.succeed(evaluateExpressionWorkerRequest(worker.posted!));
  const result = await pending;
  check(result.data[0] === 5, "worker result resolves with the expected field");
  check(worker.terminated === 1, "successful expression workers terminate after returning output");
}
throws(() => evaluateExpressionDerivedField(compiled, {
  u: { shape: [2, 2], data: Float32Array.from([1, 2, 3, 4]) },
  v: { shape: [4], data: Float32Array.from([1, 2, 3, 4]) },
}), /does not exactly match/, "equal-length but different shapes are rejected");
throws(() => evaluateExpressionDerivedField(compiled, {
  u: { shape: [2, 2], data: Float32Array.from([1, 2, 3]) },
  v: { shape: [2, 2], data: Float32Array.from([1, 2, 3, 4]) },
}), /data length/, "shape product must exactly match data length");
throws(() => evaluateExpressionDerivedField(compiled, {
  u: { shape: [2, 2], data: Float32Array.from([1, 2, 3, 4]) },
  v: { shape: [2, 2], data: Float32Array.from([1, 2, 3, 4]) },
}, { maxElements: 3 }), /too large/, "caller evaluation cap is enforced");
const singular = createExpressionDerivedField({ id: "singular", label: "Singular", unit: "1", dependencies: ["u"], formula: "1 / (u - u)" });
check(Number.isNaN(evaluateExpressionDerivedField(singular, { u: { shape: [1], data: Float32Array.of(2) } }).data[0]), "nonfinite formula results remain NaN");
const overflow = createExpressionDerivedField({ id: "overflow", label: "Overflow", unit: "1", dependencies: ["u"], formula: "u * 10^38" });
check(Number.isNaN(evaluateExpressionDerivedField(overflow, { u: { shape: [1], data: Float32Array.of(10) } }).data[0]), "finite double results that overflow Float32 become NaN");

const identityA = expressionDerivedFieldDefinitionIdentity(wind);
const identityB = expressionDerivedFieldCacheIdentity(parseExpressionDerivedField(JSON.parse(JSON.stringify(wind))));
const changed = createExpressionDerivedField({ ...wind, parameters: [{ id: "scale", value: 2 }] });
check(identityA === identityB && identityA !== expressionDerivedFieldCacheIdentity(changed), "stable cache identity binds formula dependencies units and parameter values");
const relabeled = createExpressionDerivedField({ ...wind, label: "Publication label" });
check(expressionDerivedFieldCacheIdentity(relabeled) === identityA, "display-only label does not invalidate numerical cache identity");

const storage = new MemoryStorage();
const datasetA = "dataset-alpha";
const datasetB = "dataset-beta";
const persistenceOptions = { nativeFields: fields, availableFields: fields, reservedFieldIds: new Set(["temperature"]) };
saveExpressionDerivedFields(storage, datasetA, [wind], persistenceOptions);
check(loadExpressionDerivedFields(storage, datasetA, persistenceOptions).definitions[0]?.id === wind.id, "dataset-scoped definitions persist and reload");
check(loadExpressionDerivedFields(storage, datasetB, persistenceOptions).definitions.length === 0, "another dataset cannot observe persisted definitions");
const key = expressionDerivedFieldStorageKey(datasetA);
const envelope = JSON.parse(storage.getItem(key)!);
storage.setItem(key, JSON.stringify({ ...envelope, datasetIdentity: datasetB }));
check(loadExpressionDerivedFields(storage, datasetA, persistenceOptions).definitions.length === 0 && loadExpressionDerivedFields(storage, datasetA, persistenceOptions).warnings.length === 1, "storage identity mismatch rejects the complete set");
storage.setItem(key, "not-json");
check(loadExpressionDerivedFields(storage, datasetA, persistenceOptions).definitions.length === 0, "malformed storage fails closed");
saveExpressionDerivedFields(storage, datasetA, [wind], persistenceOptions);
clearExpressionDerivedFields(storage, datasetA);
check(storage.getItem(key) === null, "dataset-local definitions can be cleared explicitly");

const many = Array.from({ length: MAX_EXPRESSION_DERIVED_FIELDS + 1 }, (_, index) => createExpressionDerivedField({
  id: `field_${index}`,
  label: `Field ${index}`,
  unit: "1",
  dependencies: ["u"],
  formula: "u",
}));
throws(() => saveExpressionDerivedFields(storage, datasetA, many), /at most/, "persistence enforces the bounded-definition limit");

const dataset = createLocalDatasetSource("cm1_netcdf", ["/case/cm1out_000001.nc", "/case/cm1out_000002.nc"]);
const step: StepInfo = {
  index: 7,
  simulationTimeS: 420,
  fields: ["u", "v"],
  grid: { nr: 2, nth: 2, nz: 1, geometryKind: "physical" },
  fieldUnits: { u: "m/s", v: "m/s" },
  publicationState: "complete",
  publicationGeneration: "cm1-record-7",
};
const provenance = buildExpressionDerivedFieldProvenance(dataset, step, wind);
check(provenance.schema === EXPRESSION_DERIVED_FIELD_PROVENANCE_SCHEMA && provenance.datasetIdentity === datasetSourceIdentity(dataset), "provenance binds source-neutral dataset identity");
check(provenance.frameIndex === 7 && provenance.publicationGeneration === "cm1-record-7" && provenance.dependencyIdentities.length === 2, "provenance binds exact frame generation and dependencies");

console.log(`expression-derived fields: ${passed} checks passed, ${failures.length} failed`);
if (failures.length) {
  failures.forEach((failure) => console.error(`FAIL ${failure}`));
  process.exit(1);
}
