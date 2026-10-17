import { parseNpy, type Npy3D, type NpyN } from "@/lib/platform/npy";
import { parseStormTrack, type StormTrackSnapshot } from "./stormTrack";
import type { FrameMotion } from "@/lib/run/frameMotion";
import { clearPlaybackFrameMetadata, registerPlaybackFrameMetadata } from "./framePrefetch";
import { isTauriRuntime } from "@/lib/platform/tauri";
import {
  clearFieldCache as clearParsedFieldCache,
  getOrLoadField,
  getOrLoadSharedAbortableField,
  makeFieldKey,
} from "@/lib/core/fieldCache";
import {
  estimateBoundedFieldWindowBytes,
  fieldLoadScheduler,
  type LoadScheduleOptions,
} from "@/lib/core/loadScheduler";
import {
  cm1HistoryMappingForCanonical,
  isNetcdfDatasetPath,
  nativeCm1Field,
  transformCm1HistoryValues,
} from "@/lib/cm1/cm1HistoryMapping";
import {
  derivedFieldCacheIdentity,
  derivedFieldDefinition,
} from "@/lib/derived/derivedFieldRegistry";
import { evaluateDerivedFieldInWorker } from "@/lib/derived/derivedFieldWorkerClient";
import {
  compileExpressionDerivedField,
  expressionDerivedFieldCacheIdentity,
  type CompiledExpressionDerivedField,
  type ExpressionDerivedFieldDefinition,
} from "@/lib/derived/expressionDerivedFields";
import { evaluateExpressionDerivedInWorker } from "@/lib/derived/expressionDerivedWorkerClient";
import {
  MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES,
  readNativeFieldBrickWindow,
  readNativeFieldWindow,
  type NativeFieldWindowRequest,
} from "@/lib/dataset/nativeFieldWindow";
import { readNetcdfFieldWindow } from "@/lib/dataset/netcdfFieldWindow";
import {
  executeExternalDerivedProvider,
  externalDerivedFieldCacheIdentity,
  parseExternalDerivedProvider,
  type ExternalDerivedProvider,
  type ExternalProviderProvenance,
} from "@/lib/derived/externalDerivedProviders";
import {
  isPackagedAtmosphericExampleUri,
  listPackagedAtmosphericExampleSteps,
  readPackagedAtmosphericExampleField,
  reducePackagedAtmosphericExampleField,
  reducePackagedAtmosphericExampleFieldAxis,
} from "@/lib/dataset/packagedAtmosphericExamples";
import { isVolumetricField } from "@/lib/plot/fieldProvenance";
import {
  validateFieldBrickHierarchy,
  type FieldBrickHierarchy,
} from "@/lib/plot/fieldBrickHierarchy";
import { invokeCommand } from "@/lib/platform/appInvoke";
import { aggregateDerivedWindow, assembleWindowedField } from "./windowedField";
import { partitionDerivedWindow } from "./derivedWindowPlanning";

const MIB = 1024 * 1024;
// Covers one conservative legacy/metadata-free field transaction. Current
// manifest-backed frames replace this fallback with a grid-derived reservation.
const DEFAULT_FIELD_RESERVATION_BYTES = 192 * MIB;
// Full-field ZFP reconstruction can simultaneously hold two history fields, a
// decoded residual/target, and the IPC/NPY staging buffer. Reserve that
// worst-case footprint for every format; raw NPY loads are over-reserved but
// remain concurrent up to the scheduler's separate three-task ceiling.
const FIELD_TRANSACTION_MULTIPLIER = 4;
const fieldByteEstimates = new Map<string, number>();
const fieldExpectedShapes = new Map<string, readonly (readonly number[])[]>();
const publishedFieldsByFrame = new Map<string, ReadonlySet<string>>();
const nativeFieldsByFrame = new Map<string, ReadonlySet<string>>();
const compressedFieldsByFrame = new Map<string, ReadonlySet<string>>();
const gridsByFrame = new Map<string, GridInfo | null>();
const netcdfFrames = new Map<string, { path: string; recordIndex: number }>();
const expressionFieldsByDataset = new Map<string, ReadonlyMap<string, CompiledExpressionDerivedField>>();
interface RegisteredExternalProviderSet {
  datasetIdentity: string;
  providers: ReadonlyMap<string, ExternalDerivedProvider>;
  frames: ReadonlyMap<number, StepInfo>;
}
const externalProvidersByDataset = new Map<string, RegisteredExternalProviderSet>();
const externalProviderProvenance = new Map<string, ExternalProviderProvenance>();
let externalFieldProvenance = new WeakMap<NpyN, ExternalProviderProvenance>();
let externalRequestSequence = 0;
let nativeWindowRequestSequence = 0;
const expressionLatestControllers = new Map<string, AbortController>();
const stormTrackReads = new Map<string, Promise<StormTrackSnapshot | null>>();

function beginExpressionLatestRequest(
  latestScope: string | undefined,
): { signal: AbortSignal | undefined; dispose: () => void } {
  if (!latestScope) return { signal: undefined, dispose: () => undefined };
  expressionLatestControllers.get(latestScope)?.abort();
  const controller = new AbortController();
  expressionLatestControllers.set(latestScope, controller);
  return {
    signal: controller.signal,
    dispose: () => {
      if (expressionLatestControllers.get(latestScope) === controller) {
        expressionLatestControllers.delete(latestScope);
      }
    },
  };
}

function linkedAbortSignal(
  ...candidates: readonly (AbortSignal | undefined)[]
): { signal: AbortSignal | undefined; dispose: () => void } {
  const signals = candidates.filter((signal): signal is AbortSignal => signal != null);
  if (signals.length <= 1) {
    return { signal: signals[0], dispose: () => undefined };
  }
  const controller = new AbortController();
  const abort = () => controller.abort();
  for (const signal of signals) {
    if (signal.aborted) {
      controller.abort();
      break;
    }
    signal.addEventListener("abort", abort, { once: true });
  }
  return {
    signal: controller.signal,
    dispose: () => {
      for (const signal of signals) signal.removeEventListener("abort", abort);
    },
  };
}

function throwIfFieldReadAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) throw new DOMException("Field read cancelled", "AbortError");
}

export function clearFieldCache(): void {
  clearPlaybackFrameMetadata();
  for (const controller of expressionLatestControllers.values()) controller.abort();
  expressionLatestControllers.clear();
  clearParsedFieldCache();
  fieldLoadScheduler.cancelQueued();
  fieldByteEstimates.clear();
  fieldExpectedShapes.clear();
  publishedFieldsByFrame.clear();
  nativeFieldsByFrame.clear();
  compressedFieldsByFrame.clear();
  gridsByFrame.clear();
  netcdfFrames.clear();
  expressionFieldsByDataset.clear();
  externalProvidersByDataset.clear();
  externalProviderProvenance.clear();
  externalFieldProvenance = new WeakMap();
  nestedHistoryRequests.clear();
}

/* Bridge to the Rust outputs commands. In browser-only mode, directory calls return null/[]. */

export interface GridInfo {
  nr?: number;
  nth?: number;
  nz?: number;
  coordinateSystem?: string;
  geometryKind?: "physical" | "index";
  staggering?: string;
  drM?: number;
  dyM?: number;
  dthetaRad?: number;
  dzM?: number;
  /** True when scalar cells use non-separable 3-D physical height arrays. */
  terrainFollowing?: boolean;
  /** Immutable lossless terrain-geometry content identity for manifest v7. */
  terrainGeometryIdentity?: string;
  /** Exact x/y center of storage cell (0,0) for terrain-following output. */
  horizontalScalarOriginM?: [number, number];
  /** Exact separable scalar-center coordinates in metres. Present for imported
   * datasets whose axes are physically declared, including stretched grids. */
  scalarCoordinatesM?: [number[], number[], number[]];
  scalarOriginM?: [number, number, number];
  physicalBoundsM?: AxisBounds;
  validIndexRange?: IndexBounds;
  haloWidth?: IndexBounds;
}

export interface AxisBounds {
  lower: [number, number, number];
  upper: [number, number, number];
}

export interface IndexBounds {
  lower: [number, number, number];
  upper: [number, number, number];
}

export interface StepInfo {
  index: number;
  simulationTimeS: number | null;
  fields: string[];
  grid: GridInfo | null;
  /** field name -> physical units, from the run manifest's field catalog
   * (emitted by the simulator from field_contract.cpp). Empty for manifests
   * that predate the catalog. Used to label colorbars/axes; see lib/fieldUnits. */
  fieldUnits: Record<string, string>;
  /** Exact manifest-declared lossy storage contract; absent/empty is lossless
   * or legacy, never an inferred tolerance. */
  fieldCompression?: Record<string, FieldCompressionInfo>;
  fieldBrickHierarchies?: Record<string, FieldBrickHierarchyInfo>;
  compressionExecution?: CompressionExecutionInfo | null;
  /** Source-neutral field origin. Native data always takes precedence over a
   * Studio definition with the same canonical ID. Older fixtures may omit it. */
  fieldOrigins?: Record<string, "native" | "studio_derived" | "user_expression" | "external_provider">;
  /** Validated simulator publication state. `legacy` is an explicit bridge
   * classification for runs with no nimbus_output.json contract; current
   * incomplete/invalid frames are never selectable. */
  publicationState: StepPublicationState;
  /** Unique generation for a current simulator publication. Null only for the
   * deliberate legacy path. Part of every field/cache/request identity. */
  publicationGeneration: string | null;
  /**
   * Domain-motion frame provenance (docs/domain.md "Output, ZFP, plotting, and
   * diagnostics"): Earth origin, grid origin, frame velocity, motion mode, and
   * cumulative shift.
   *
   * Optional because every run written before the domain-motion contract omits
   * it, and those runs were stationary — absence is therefore read as
   * stationary rather than unknown. Declared here now so the plots that combine
   * frames by grid cell can fail closed the moment a translating run appears,
   * rather than silently smearing a swath. See lib/run/frameMotion.ts.
   */
  motion?: FrameMotion | null;
}

export interface FieldCompressionInfo {
  codec: string;
  mode: string;
  modeParameter: number;
  modeParameterUnits: string;
  errorContract: string;
  sparseThreshold: number;
  float16Prefilter: boolean;
  decodedDtype: string;
}

export interface FieldBrickHierarchyInfo {
  path: string;
  contentSha256: string;
  schemaVersion: number;
  nativeBrickShape: [number, number, number];
  extremaSemantics: "stored_values" | "decoded_value_bounds" | "pre_storage_source";
  extremaAbsolutePadding: number;
  cullingSafe: boolean;
}

export interface CompressionExecutionInfo {
  fieldExecutionPolicy: "serial" | "field_parallel";
  fieldWorkers: number;
  fieldScratchBudgetMib: number;
}

export type StepPublicationState =
  | "complete"
  | "incomplete"
  | "invalid_manifest"
  | "legacy";

/** Show the native directory picker. Returns null if the user cancels. */
export async function openRunDirectory(): Promise<string | null> {
  if (!isTauriRuntime()) {
    console.info("[mock] openRunDirectory called outside Tauri");
    return null;
  }
  const { open } = await import("@tauri-apps/plugin-dialog");
  const result = await open({
    directory: true,
    multiple: false,
    title: "Open Nimbus run directory",
  });
  if (result == null) return null;
  return typeof result === "string" ? result : result;
}

/** List step_NNNNNN/ subdirectories of `runDir`, sorted by index. */
export async function listSteps(runDir: string): Promise<StepInfo[]> {
  let steps: StepInfo[];
  if (isPackagedAtmosphericExampleUri(runDir)) {
    steps = listPackagedAtmosphericExampleSteps(runDir);
  } else {
    if (!isTauriRuntime()) return [];
    const raw = await invokeCommand<RawStepInfo[]>("list_step_dirs", { runDir });
    steps = raw.map(normalizeStep);
  }
  if (fieldByteEstimates.size > 50_000) {
    fieldByteEstimates.clear();
    fieldExpectedShapes.clear();
    publishedFieldsByFrame.clear();
    nativeFieldsByFrame.clear();
  }
  registerListedStepMetadata(runDir, steps, { fullSnapshot: true });
  return steps;
}

export interface NestedHistoryLevel {
  /** Canonical path identity, e.g. `child_0000/child_0002`. */
  levelId: string;
  /** `root` for a direct child, otherwise the parent's full level identity. */
  parentLevelId: string;
  /** One for a direct child; discovery rejects unbounded depth. */
  depth: number;
  /** Validated native history directory used by the ordinary field readers. */
  runDir: string;
  /** Frames validated against the parent publication generation. */
  steps: StepInfo[];
}

interface RawNestedHistoryLevel {
  levelId: string;
  parentLevelId: string;
  depth: number;
  runDir: string;
  steps: RawStepInfo[];
}

const nestedHistoryRequests = new Map<string, Promise<NestedHistoryLevel[]>>();

/**
 * Discover simulator-published refinement histories belonging to a Nimbus run.
 * Child manifests inherit the root generation contract; the Rust bridge
 * validates that relationship before any level becomes visible here.
 */
export async function listNestedHistoryLevels(
  runDir: string,
  expectedGeneration: string | null = null,
  stepIndex: number | null = null,
): Promise<NestedHistoryLevel[]> {
  if (isPackagedAtmosphericExampleUri(runDir) || !isTauriRuntime()) return [];
  const key = `${runDir}\u0000${expectedGeneration ?? "legacy"}\u0000${stepIndex ?? "all"}`;
  const cached = nestedHistoryRequests.get(key);
  if (cached) return cached;
  if (nestedHistoryRequests.size >= 64) nestedHistoryRequests.clear();
  const request = invokeCommand<RawNestedHistoryLevel[]>("list_nested_history_levels", { runDir, stepIndex })
    .then((raw) => {
      const levels = raw.map((level) => ({
        levelId: level.levelId,
        parentLevelId: level.parentLevelId,
        depth: level.depth,
        runDir: level.runDir,
        steps: level.steps.map(normalizeStep),
      }));
      // Register each child under its own path so ordinary bounded reads retain
      // the same shape, publication, cache, and memory-admission checks as root.
      for (const level of levels) {
        registerListedStepMetadata(level.runDir, level.steps);
      }
      // A targeted miss may mean the child product is still completing behind
      // an already-visible root frame. Do not pin that transient absence in
      // the dedupe cache; the hook performs a short bounded publication grace.
      if (stepIndex != null && !levels.some((level) => level.steps.length > 0)) {
        nestedHistoryRequests.delete(key);
      }
      return levels;
    })
    .catch((error) => {
      nestedHistoryRequests.delete(key);
      throw error;
    });
  nestedHistoryRequests.set(key, request);
  return request;
}

/** Register adapter-validated frame metadata for callers that already own a
 * StepInfo table (direct renderer hosts and integration harnesses). Ordinary
 * Plot Studio discovery calls this automatically. */
export function registerListedStepMetadata(runDir: string, steps: readonly StepInfo[], options: { fullSnapshot?: boolean } = {}): void {
  registerPlaybackFrameMetadata(runDir, steps, options.fullSnapshot === true);
  for (const step of steps) {
    const key = fieldEstimateKey(runDir, step.index, step.publicationGeneration);
    nativeFieldsByFrame.set(key, new Set(step.fields.filter(field =>
      !step.fieldOrigins?.[field] || step.fieldOrigins[field] === "native")));
    compressedFieldsByFrame.set(key, new Set(Object.entries(step.fieldCompression ?? {})
      .filter(([, compression]) => compression.codec === "zfp").map(([field]) => field)));
    gridsByFrame.set(key, step.grid);
    const grid = step.grid;
    if (!grid?.nr || !grid.nth || !grid.nz) continue;
    const elements = grid.nr * grid.nth * grid.nz;
    if (Number.isSafeInteger(elements) && elements > 0) {
      fieldByteEstimates.set(
        key,
        elements * Float32Array.BYTES_PER_ELEMENT * FIELD_TRANSACTION_MULTIPLIER,
      );
    }
    if (step.publicationGeneration !== null) {
      publishedFieldsByFrame.set(key, new Set(step.fields));
      for (const field of step.fields) {
        fieldExpectedShapes.set(
          makeFieldKey(runDir, step.index, field, step.publicationGeneration),
          [isVolumetricField(field)
            ? [grid.nr, grid.nth, grid.nz]
            : [grid.nr, grid.nth]],
        );
      }
    }
  }
}

/** Plot budgets include codec scratch; explicit point-sampling budgets remain strict. */
export function plotFieldWindows(
  runDir: string, stepIndex: number | null, fieldName: string | null, publicationGeneration: string | null,
  windows: readonly FieldLoadWindow[] | null | undefined,
): readonly FieldLoadWindow[] | null {
  if (!windows || stepIndex == null || !fieldName
      || !compressedFieldsByFrame.get(fieldEstimateKey(runDir, stepIndex, publicationGeneration))?.has(fieldName)) return windows ?? null;
  return windows.map(window => {
    if (window.extent.length !== 3 || window.stride.length !== 3 || window.aggregation === "point_sample"
        || [...window.extent, ...window.stride].some(value => !Number.isSafeInteger(value) || value < 1)) return window;
    const sourceBytes = window.extent.reduce((bytes, size) => bytes * size, 4);
    const elements = window.extent.reduce((count, size, axis) => count * Math.ceil(size / window.stride[axis]), 1);
    const scratch = window.stride.some(stride => stride !== 1) ? 21 * elements : 0;
    const needed = sourceBytes + Math.max(10 * MIB, scratch);
    if (!Number.isSafeInteger(needed)) return window;
    return { ...window, maxSourceBytes: Math.min(MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES,
      Math.max(window.maxSourceBytes ?? 0, needed)) };
  });
}

export interface FieldLoadOptions {
  /** Stable hook/widget lane whose queued older frames may be superseded. */
  latestScope?: string;
  /** Caller-owned cancellation. Native window reads stop cooperatively; full-field
   * readers discard a result at their next safe boundary. */
  signal?: AbortSignal;
  /** Background reads may use only the scheduler slots left after reserving one for interaction. */
  priority?: "interactive" | "background";
  /** Window tiles are assembly scratch and must not double the resident cache footprint. */
  cacheRetention?: "resident" | "transient";
  /** A transferable result bypasses resident/shared caches so the caller may
   * detach its ArrayBuffer when handing ownership to a worker. */
  ownership?: "shared" | "transferable";
  /** Native Nimbus or netCDF spatial subset/LOD. Derived fields evaluate the
   * exact raw tile first, then apply this display aggregation. External
   * providers must explicitly declare chunk-aware, match-input semantics. */
  window?: Omit<NativeFieldWindowRequest, "schemaVersion" | "requestId">;
  /** Durable hierarchy payload selected by a validated brick planner. */
  brickHierarchyLevel?: number;
}

/** The window a caller supplies; the reader adds schema and request identity. */
export type FieldLoadWindow = NonNullable<FieldLoadOptions["window"]>;

function registeredExternalFieldRank(
  runDir: string,
  stepIndex: number,
  fieldName: string,
  publicationGeneration: string | null,
): number | null {
  const external = externalProvidersByDataset.get(runDir)?.providers.get(fieldName);
  if (!external) return null;
  const policy = external.output.shapePolicy;
  if (policy.kind === "exact") return policy.dimensions.length;
  const input = external.inputs.find(({ alias }) => alias === policy.inputAlias);
  return input ? registeredFieldRank(runDir, stepIndex, input.fieldId, publicationGeneration) : null;
}

/** Return the exact registered rank when an adapter published one unambiguous
 * shape for this field. This is metadata-only and never triggers a field read. */
export function registeredFieldRank(
  runDir: string,
  stepIndex: number,
  fieldName: string,
  publicationGeneration: string | null,
): number | null {
  const shapes = fieldExpectedShapes.get(
    makeFieldKey(runDir, stepIndex, fieldName, publicationGeneration),
  );
  if (!shapes || shapes.length === 0) {
    const declared = derivedFieldDefinition(fieldName)?.rank
      ?? expressionFieldsByDataset.get(runDir)?.get(fieldName)?.definition.rank
      ?? registeredExternalFieldRank(runDir, stepIndex, fieldName, publicationGeneration);
    if (declared != null) return declared;
    const native = nativeFieldsByFrame.get(
      fieldEstimateKey(runDir, stepIndex, publicationGeneration),
    );
    if (native?.has(fieldName)) return isVolumetricField(fieldName) ? 3 : 2;
    return null;
  }
  const ranks = new Set(shapes.map((shape) => shape.length));
  return ranks.size === 1 ? ranks.values().next().value ?? null : null;
}

/** Install the already-validated, dataset-scoped expression registry used by
 * lazy reads. Numerical definition identity is part of each cache key. */
export function registerExpressionDerivedFields(
  runDir: string,
  definitions: readonly ExpressionDerivedFieldDefinition[],
): void {
  expressionFieldsByDataset.set(runDir, new Map(definitions.map((definition) => [
    definition.id,
    compileExpressionDerivedField(definition),
  ])));
}

/** Install enabled, already-validated external providers for one dataset.
 * Registration alone executes nothing; reads still pass the adapter permission,
 * integrity, shape, unit, and provenance gates. */
export function registerExternalDerivedProviders(
  runDir: string,
  datasetIdentity: string,
  steps: readonly StepInfo[],
  providers: readonly ExternalDerivedProvider[],
): void {
  for (const step of steps) {
    gridsByFrame.set(
      fieldEstimateKey(runDir, step.index, step.publicationGeneration),
      step.grid,
    );
  }
  externalProvidersByDataset.set(runDir, {
    datasetIdentity,
    providers: new Map(providers.map((provider) => {
      const parsed = parseExternalDerivedProvider(provider);
      return [parsed.output.fieldId, parsed] as const;
    })),
    frames: new Map(steps.map((step) => [step.index, step])),
  });
}

export function externalDerivedFieldProvenance(
  runDir: string,
  stepIndex: number,
  fieldName: string,
  publicationGeneration: string | null,
): ExternalProviderProvenance | null {
  return externalProviderProvenance.get(makeFieldKey(runDir, stepIndex, fieldName, publicationGeneration)) ?? null;
}

function fieldEstimateKey(
  runDir: string,
  stepIndex: number,
  publicationGeneration: string | null,
): string {
  return JSON.stringify([runDir, publicationGeneration, stepIndex]);
}

export function registerNetcdfFrame(
  runDir: string,
  stepIndex: number,
  publicationGeneration: string,
  path: string,
  recordIndex: number,
  shape: readonly number[],
  fieldShapes: Readonly<Record<string, readonly number[]>>,
  nativeFields: readonly string[] = Object.keys(fieldShapes),
): void {
  const key = fieldEstimateKey(runDir, stepIndex, publicationGeneration);
  netcdfFrames.set(key, { path, recordIndex });
  const elements = shape.reduce((total, length) => total * length, 1);
  if (Number.isSafeInteger(elements) && elements > 0) {
    fieldByteEstimates.set(key, elements * Float32Array.BYTES_PER_ELEMENT * FIELD_TRANSACTION_MULTIPLIER);
    for (const [field, fieldShape] of Object.entries(fieldShapes)) {
      fieldExpectedShapes.set(
        makeFieldKey(runDir, stepIndex, field, publicationGeneration),
        [fieldShape],
      );
    }
  }
  publishedFieldsByFrame.set(key, new Set(nativeFields));
  nativeFieldsByFrame.set(key, new Set(nativeFields));
}

function registeredGrid(
  runDir: string,
  stepIndex: number,
  publicationGeneration: string | null,
): GridInfo | null {
  return gridsByFrame.get(fieldEstimateKey(runDir, stepIndex, publicationGeneration)) ?? null;
}

export function fieldDefinitionIdentity(
  runDir: string,
  stepIndex: number | null,
  fieldName: string | null,
  publicationGeneration: string | null,
): string {
  if (!fieldName) return "";
  if (stepIndex != null && nativeFieldsByFrame.get(fieldEstimateKey(runDir, stepIndex, publicationGeneration))?.has(fieldName)) return fieldName;
  if (derivedFieldDefinition(fieldName)) return derivedFieldCacheIdentity(fieldName);
  const expression = expressionFieldsByDataset.get(runDir)?.get(fieldName);
  if (expression) return expressionDerivedFieldCacheIdentity(expression.definition);
  const external = externalProvidersByDataset.get(runDir);
  const provider = external?.providers.get(fieldName);
  return provider ? externalDerivedFieldCacheIdentity(provider, external!.datasetIdentity) : fieldName;
}

/**
 * Read a field of any rank (2D surface diagnostics or 3D volumes).
 * Results are cached by (runDir, generation, step, field) and must be treated as read-only.
 */
export async function readFieldAny(
  runDir: string,
  stepIndex: number,
  fieldName: string,
  expectedGeneration: string | null,
  options: FieldLoadOptions = {},
): Promise<NpyN> {
  throwIfFieldReadAborted(options.signal);
  const packagedExample = isPackagedAtmosphericExampleUri(runDir);
  if (!packagedExample && !isTauriRuntime()) {
    throw new Error("readFieldAny requires the Tauri runtime");
  }
  const frameKey = fieldEstimateKey(runDir, stepIndex, expectedGeneration);
  const isNative = nativeFieldsByFrame.get(frameKey)?.has(fieldName) === true;
  const derivedDefinition = isNative ? null : derivedFieldDefinition(fieldName);
  const expressionDefinition = isNative || derivedDefinition
    ? null
    : expressionFieldsByDataset.get(runDir)?.get(fieldName) ?? null;
  const externalSet = isNative || derivedDefinition || expressionDefinition
    ? null
    : externalProvidersByDataset.get(runDir) ?? null;
  const externalProvider = externalSet?.providers.get(fieldName) ?? null;
  const cacheFieldName = derivedDefinition
    ? derivedFieldCacheIdentity(fieldName)
    : expressionDefinition
      ? expressionDerivedFieldCacheIdentity(expressionDefinition.definition)
      : externalProvider
        ? externalDerivedFieldCacheIdentity(externalProvider, externalSet!.datasetIdentity)
        : fieldName;
  const cacheIdentity = options.window
    ? JSON.stringify([cacheFieldName, "window-v1", options.brickHierarchyLevel ?? 0, options.window])
    : cacheFieldName;
  const fieldKey = makeFieldKey(runDir, stepIndex, cacheIdentity, expectedGeneration);
  const loadField = async (
    consumerSignal = options.signal,
    latestScope = options.latestScope,
  ) => {
      throwIfFieldReadAborted(consumerSignal);
      // Minted here, not at call entry: a cache hit or in-flight dedup never
      // reaches this callback, and must not abort another frame's live read.
      const expressionRequest = beginExpressionLatestRequest(latestScope);
      const linked = linkedAbortSignal(expressionRequest.signal, consumerSignal);
      const operationSignal = linked.signal;
      try {
        throwIfFieldReadAborted(operationSignal);
      if (options.window && externalProvider
          && (!externalProvider.capabilities.includes("chunk-aware")
            || externalProvider.output.shapePolicy.kind !== "match-input")) {
        throw new Error(
          `External provider '${externalProvider.id}' cannot serve a bounded plot window. `
            + "Add the chunk-aware capability and a match-input output shape policy, or publish the result.",
        );
      }
      const pointDerived = derivedDefinition != null && options.window?.aggregation === "point_sample";
      if (options.window && !pointDerived && (derivedDefinition || expressionDefinition || externalProvider)) {
        const tiles = partitionDerivedWindow(options.window,
          registeredGrid(runDir, stepIndex, expectedGeneration)?.terrainFollowing === true);
        if (tiles.length > 1) {
          const chunks: NpyN[] = [];
          for (const window of tiles) {
            throwIfFieldReadAborted(operationSignal);
            chunks.push(await readFieldAny(runDir, stepIndex, fieldName, expectedGeneration, {
              window, signal: operationSignal, priority: options.priority, cacheRetention: "transient",
            }));
          }
          throwIfFieldReadAborted(operationSignal);
          return assembleWindowedField(chunks, tiles);
        }
      }
      const rawDerivedWindow = pointDerived ? options.window! : options.window ? {
        ...options.window,
        aggregation: "cell_mean" as const,
        stride: options.window.stride.map(() => 1),
        maxOutputBytes: options.window.extent.reduce(
          (bytes, count) => bytes * count,
          (registeredGrid(runDir, stepIndex, expectedGeneration)?.terrainFollowing === true ? 36 : 4),
        ),
        maxSourceBytes: registeredGrid(runDir, stepIndex, expectedGeneration)?.terrainFollowing === true
          ? MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES
          : options.window.extent.reduce(
              (bytes, count) => bytes * count,
              Float32Array.BYTES_PER_ELEMENT,
            ),
      } : null;
      const dependencyWindow = (field: string) => rawDerivedWindow
        ? plotFieldWindows(runDir, stepIndex, field, expectedGeneration, [rawDerivedWindow])![0]
        : null;
      if (derivedDefinition) {
        const dependencies = await Promise.all(derivedDefinition.dependencies.map(async ({ field }) => [
          field,
          await readFieldAny(
            runDir,
            stepIndex,
            field,
            expectedGeneration,
            rawDerivedWindow
              ? { window: dependencyWindow(field)!, signal: operationSignal, priority: options.priority, cacheRetention: "transient" }
              : { signal: operationSignal, priority: options.priority },
          ),
        ] as const));
        // Slabbed onto a worker: evaluating a 42M-cell theta_e inline blocked the UI for
        // about three seconds per frame, while user expressions beside it already ran off-thread.
        const derived = await evaluateDerivedFieldInWorker(
          fieldName,
          Object.fromEntries(dependencies),
          { signal: operationSignal },
        );
        if (!options.window || pointDerived) return derived;
        const sourceShape = dependencies[0][1].windowGeometry?.sourceShape;
        if (!sourceShape) throw new Error("Derived raw window did not preserve its source shape");
        return aggregateDerivedWindow(derived, options.window, sourceShape);
      }
      if (expressionDefinition) {
        const dependencies = await Promise.all(expressionDefinition.definition.dependencies.map(async (field) => [
          field,
          await readFieldAny(
            runDir,
            stepIndex,
            field,
            expectedGeneration,
            rawDerivedWindow
              ? { window: dependencyWindow(field)!, signal: operationSignal, priority: options.priority, cacheRetention: "transient" }
              : { signal: operationSignal, priority: options.priority },
          ),
        ] as const));
        const expression = await evaluateExpressionDerivedInWorker(
          expressionDefinition,
          Object.fromEntries(dependencies),
          { signal: operationSignal },
        );
        if (!options.window) return expression;
        const geometry = dependencies[0][1].windowGeometry;
        const sourceShape = geometry?.sourceShape;
        if (!sourceShape) throw new Error("Expression raw window did not preserve its source shape");
        return aggregateDerivedWindow(
          { ...expression, windowGeometry: geometry },
          options.window,
          sourceShape,
        );
      }
      if (externalProvider && externalSet) {
        const frame = externalSet.frames.get(stepIndex);
        if (!frame) throw new Error(`External provider frame ${stepIndex} is not registered`);
        const dependencies = await Promise.all(externalProvider.inputs.map(async ({ fieldId, alias }) => [
          alias,
          {
            field: await readFieldAny(
              runDir,
              stepIndex,
              fieldId,
              expectedGeneration,
              rawDerivedWindow
                ? { window: dependencyWindow(fieldId)!, signal: operationSignal, priority: options.priority, cacheRetention: "transient" }
                : { signal: operationSignal, priority: options.priority },
            ),
            unit: frame.fieldUnits[fieldId] ?? null,
            identity: makeFieldKey(runDir, stepIndex, fieldId, expectedGeneration),
          },
        ] as const));
        if (externalProvider.kind !== "python") {
          throw new Error(`Native plugin provider '${externalProvider.id}' is registered but no plugin-host adapter is connected`);
        }
        const { createPyodideDerivedProviderAdapter } = await import("@/lib/derived/pyodideDerivedProviderAdapter");
        const controller = operationSignal ? null : new AbortController();
        const result = await executeExternalDerivedProvider(
          externalProvider,
          createPyodideDerivedProviderAdapter(),
          {
            requestId: `external:${++externalRequestSequence}:${stepIndex}:${externalProvider.id}`,
            datasetIdentity: externalSet.datasetIdentity,
            frame: {
              index: stepIndex,
              simulationTimeS: frame.simulationTimeS,
              publicationGeneration: expectedGeneration,
            },
            grid: frame.grid,
            inputs: Object.fromEntries(dependencies),
            grantedPermissions: new Set(externalProvider.permissions),
            signal: operationSignal ?? controller!.signal,
          },
        );
        if (!options.window) {
          externalFieldProvenance.set(result.field, result.provenance);
          return result.field;
        }
        const reference = dependencies[0][1].field;
        const sourceShape = reference.windowGeometry?.sourceShape;
        if (!sourceShape) throw new Error("External-provider raw window did not preserve its source shape");
        const aggregated = aggregateDerivedWindow(
          { ...result.field, windowGeometry: reference.windowGeometry },
          options.window,
          sourceShape,
        );
        externalFieldProvenance.set(aggregated, result.provenance);
        return aggregated;
      }
      const publishedFields = publishedFieldsByFrame.get(frameKey);
      if (publishedFields && !publishedFields.has(fieldName)) {
        throw new Error(
          `Field '${fieldName}' is not published by the selected step manifest`,
        );
      }
      const estimatedBytes = options.window
        ? estimateBoundedFieldWindowBytes(
            options.window,
            registeredGrid(runDir, stepIndex, expectedGeneration)?.terrainFollowing === true,
          )
        : fieldByteEstimates.get(frameKey) ?? DEFAULT_FIELD_RESERVATION_BYTES;
      const schedule: LoadScheduleOptions = {
        estimatedBytes,
        latestScope,
        priority: options.priority,
        signal: operationSignal,
      };
      const netcdfFrame = netcdfFrames.get(frameKey);
      const netcdfSource = netcdfFrame != null || isNetcdfDatasetPath(runDir);
      if (packagedExample) {
        const packaged = await readPackagedAtmosphericExampleField(
          runDir,
          stepIndex,
          fieldName,
          expectedGeneration,
          options.window,
        );
        throwIfFieldReadAborted(operationSignal);
        return packaged;
      }
      if (options.window) {
        const requestId = `window:${++nativeWindowRequestSequence}:${stepIndex}:${fieldName}`;
        if ((options.brickHierarchyLevel ?? 0) > 0) {
          if (!expectedGeneration) {
            throw new Error("Durable brick LOD reads require a publication generation");
          }
          const window = await fieldLoadScheduler.schedule(schedule, async () =>
            readNativeFieldBrickWindow({
              runDir,
              stepIndex,
              fieldName,
              expectedGeneration,
              level: options.brickHierarchyLevel!,
              request: {
                schemaVersion: 2,
                requestId,
                ...options.window!,
              },
            }, operationSignal));
          throwIfFieldReadAborted(operationSignal);
          return {
            shape: window.header.outputShape,
            data: window.data,
            windowGeometry: {
              aggregation: window.header.aggregation,
              weighting: window.header.weighting,
              sampleLocation: window.header.sampleLocation,
              centroidsM: window.centroidsM,
              volumesM3: window.volumesM3,
              sourceShape: [...window.header.sourceShape],
              origin: [...window.header.origin],
              extent: [...window.header.extent],
              stride: [...window.header.stride],
              sourceFormat: window.header.sourceFormat,
            },
          };
        }
        if (netcdfSource) {
          const variable = nativeCm1Field(fieldName);
          if (!variable) throw new Error(`No qualified CM1 history mapping for '${fieldName}'`);
          const window = await fieldLoadScheduler.schedule(schedule, async () => readNetcdfFieldWindow({
            path: netcdfFrame?.path ?? runDir,
            variable,
            recordIndex: netcdfFrame?.recordIndex ?? stepIndex,
            request: {
              schemaVersion: 2,
              requestId,
              ...options.window!,
            },
          }, operationSignal));
          throwIfFieldReadAborted(operationSignal);
          return {
            shape: window.header.outputShape,
            data: window.data,
            windowGeometry: {
              aggregation: window.header.aggregation,
              weighting: window.header.weighting,
              sampleLocation: window.header.sampleLocation,
              centroidsM: null,
              volumesM3: null,
              sourceShape: [...window.header.sourceShape],
              origin: [...window.header.origin],
              extent: [...window.header.extent],
              stride: [...window.header.stride],
              sourceFormat: window.header.sourceFormat,
            },
          };
        }
        const window = await fieldLoadScheduler.schedule(schedule, async () => readNativeFieldWindow({
          runDir,
          stepIndex,
          fieldName,
          expectedGeneration: expectedGeneration ?? undefined,
          request: {
            schemaVersion: 2,
            requestId,
            ...options.window!,
          },
        }, operationSignal));
        throwIfFieldReadAborted(operationSignal);
        return {
          shape: window.header.outputShape,
          data: window.data,
          windowGeometry: {
            aggregation: window.header.aggregation,
            weighting: window.header.weighting,
            sampleLocation: window.header.sampleLocation,
            centroidsM: window.centroidsM,
            volumesM3: window.volumesM3,
            sourceShape: [...window.header.sourceShape],
            origin: [...window.header.origin],
            extent: [...window.header.extent],
            stride: [...window.header.stride],
            sourceFormat: window.header.sourceFormat,
          },
        };
      }
      const buffer = await fieldLoadScheduler.schedule(schedule, async () => {
        throwIfFieldReadAborted(operationSignal);
        if (netcdfSource) {
          const variable = nativeCm1Field(fieldName);
          if (!variable) throw new Error(`No qualified CM1 history mapping for '${fieldName}'`);
          return invokeCommand<ArrayBuffer>("read_netcdf_field", {
            path: netcdfFrame?.path ?? runDir,
            variable,
            recordIndex: netcdfFrame?.recordIndex ?? stepIndex,
          });
        }
        return invokeCommand<ArrayBuffer>("read_field_bytes", {
          runDir,
          stepIndex,
          fieldName,
          expectedGeneration,
        });
      });
      throwIfFieldReadAborted(operationSignal);
      const expectedShapes = fieldExpectedShapes.get(
        makeFieldKey(runDir, stepIndex, fieldName, expectedGeneration),
      ) ?? fieldExpectedShapes.get(frameKey);
      const parsed = parseNpy(buffer, { expectedShapes });
      const mapping = netcdfSource ? cm1HistoryMappingForCanonical(fieldName) : null;
      if (!mapping || mapping.valueScale === 1) return parsed;
      return {
        ...parsed,
        data: transformCm1HistoryValues(fieldName, parsed.data),
      };
      } finally {
        linked.dispose();
        expressionRequest.dispose();
      }
  };
  const onCacheHit = options.latestScope
    ? () => fieldLoadScheduler.supersede(options.latestScope!)
    : undefined;
  const loaded = options.ownership === "transferable"
    ? await loadField(options.signal)
    : options.signal && options.window
    ? await getOrLoadSharedAbortableField(
        fieldKey,
        // Shared native work is governed by consumer reference counting, not
        // by one arbitrary hook lane's latest-wins scope.
        (sharedSignal) => loadField(sharedSignal, undefined),
        options.signal,
        onCacheHit,
        { retain: options.cacheRetention !== "transient" },
      )
    : await getOrLoadField(
        fieldKey,
        () => loadField(),
        onCacheHit,
        {
          shareInFlight: options.signal == null,
          joinInFlight: true,
          retain: options.cacheRetention !== "transient",
        },
      );
  throwIfFieldReadAborted(options.signal);
  const provenance = externalFieldProvenance.get(loaded);
  if (provenance) externalProviderProvenance.set(makeFieldKey(runDir, stepIndex, fieldName, expectedGeneration), provenance);
  return loaded;
}

/** Read + parse a single 3D field at the given step (cached; read-only). */
export async function readField(
  runDir: string,
  stepIndex: number,
  fieldName: string,
  expectedGeneration: string | null,
  options: FieldLoadOptions = {},
): Promise<Npy3D> {
  const arr = await readFieldAny(
    runDir,
    stepIndex,
    fieldName,
    expectedGeneration,
    options,
  );
  if (arr.shape.length !== 3) {
    // Rank-2 fields (CAPE, CIN, composite_reflectivity, UH…) are an ordinary picker choice, not a corrupt file.
    const kind = arr.shape.length === 2 ? "a single-level diagnostic" : `rank ${arr.shape.length}`;
    throw new Error(
      `'${fieldName}' is ${kind} (${arr.shape.join("×")}), not a volume.`
      + " Single-level fields render on the Plan views and Swaths tabs.",
    );
  }
  return {
    shape: arr.shape as [number, number, number],
    data: arr.data,
    ...(arr.windowGeometry ? { windowGeometry: arr.windowGeometry } : {}),
  };
}

export async function readFieldBrickHierarchy(
  runDir: string,
  step: Pick<StepInfo, "index" | "publicationGeneration" | "grid" | "fieldBrickHierarchies">,
  fieldName: string,
): Promise<FieldBrickHierarchy> {
  const declared = step.fieldBrickHierarchies?.[fieldName];
  if (!declared) throw new Error(`'${fieldName}' has no published brick hierarchy.`);
  if (!isTauriRuntime()) throw new Error("Field brick hierarchies require the native runtime.");
  const raw = await invokeCommand<unknown>("read_field_brick_hierarchy", {
    runDir,
    stepIndex: step.index,
    fieldName,
    expectedGeneration: step.publicationGeneration,
  });
  const validRange = step.grid?.validIndexRange;
  const shape = validRange
    ? validRange.upper.map((upper, axis) => upper - validRange.lower[axis] + 1)
    : step.grid?.nr != null && step.grid.nth != null && step.grid.nz != null
      ? [step.grid.nr, step.grid.nth, step.grid.nz]
      : undefined;
  const sourceIndexOffset = validRange?.lower ?? (shape ? [0, 0, 0] : undefined);
  const validated = validateFieldBrickHierarchy(raw, {
    field: fieldName,
    publicationGeneration: step.publicationGeneration ?? undefined,
    sourceShape: shape,
    sourceIndexOffset,
  });
  if (!validated.ok) throw new Error(`Field brick hierarchy failed validation: ${validated.reason}`);
  if (validated.hierarchy.levels[0].brickShape.join("x")
      !== declared.nativeBrickShape.join("x")
      || validated.hierarchy.schemaVersion !== declared.schemaVersion
      || validated.hierarchy.cullingSafe !== declared.cullingSafe
      || validated.hierarchy.extremaSemantics !== declared.extremaSemantics
      || validated.hierarchy.extremaAbsolutePadding !== declared.extremaAbsolutePadding) {
    throw new Error("Field brick hierarchy disagrees with its completion manifest.");
  }
  return validated.hierarchy;
}

/** Read block-aligned bounded tiles sequentially and assemble one C-order volume. */
export async function readFieldAnyWindows(
  runDir: string,
  stepIndex: number,
  fieldName: string,
  expectedGeneration: string | null,
  windows: readonly FieldLoadWindow[],
  options: {
    latestScope?: string;
    signal?: AbortSignal;
    priority?: "interactive" | "background";
    brickHierarchyLevel?: number;
    cacheRetention?: "resident" | "transient";
  } = {},
): Promise<NpyN> {
  if (windows.length === 0) throw new Error("A bounded field read requires at least one tile");
  const frameKey = fieldEstimateKey(runDir, stepIndex, expectedGeneration);
  const isNative = nativeFieldsByFrame.get(frameKey)?.has(fieldName) === true;
  const derivedDefinition = isNative ? null : derivedFieldDefinition(fieldName);
  const expressionDefinition = isNative || derivedDefinition
    ? null
    : expressionFieldsByDataset.get(runDir)?.get(fieldName) ?? null;
  const externalSet = isNative || derivedDefinition || expressionDefinition
    ? null
    : externalProvidersByDataset.get(runDir) ?? null;
  const externalProvider = externalSet?.providers.get(fieldName) ?? null;
  const cacheFieldName = derivedDefinition
    ? derivedFieldCacheIdentity(fieldName)
    : expressionDefinition
      ? expressionDerivedFieldCacheIdentity(expressionDefinition.definition)
      : externalProvider
        ? externalDerivedFieldCacheIdentity(externalProvider, externalSet!.datasetIdentity)
        : fieldName;
  const assemblyIdentity = JSON.stringify([
    cacheFieldName, "window-assembly-v1", options.brickHierarchyLevel ?? 0, windows,
  ]);
  const assemblyKey = makeFieldKey(runDir, stepIndex, assemblyIdentity, expectedGeneration);
  const loadAssembly = async (signal = options.signal) => {
      const chunks: NpyN[] = [];
      for (const window of windows) {
        if (signal?.aborted) {
          throw new DOMException("Field read cancelled", "AbortError");
        }
        chunks.push(await readFieldAny(runDir, stepIndex, fieldName, expectedGeneration, {
          signal,
          window,
          priority: options.priority,
          cacheRetention: "transient",
          brickHierarchyLevel: options.brickHierarchyLevel,
        }));
      }
      if (signal?.aborted) {
        throw new DOMException("Field read cancelled", "AbortError");
      }
      const assembled = assembleWindowedField(chunks, windows);
      const provenance = externalFieldProvenance.get(chunks[chunks.length - 1]);
      if (provenance) externalFieldProvenance.set(assembled, provenance);
      return assembled;
  };
  const onAssemblyCacheHit = options.latestScope
    ? () => fieldLoadScheduler.supersede(options.latestScope!)
    : undefined;
  const assembled = options.signal
    ? await getOrLoadSharedAbortableField(
        assemblyKey,
        (sharedSignal) => loadAssembly(sharedSignal),
        options.signal,
        onAssemblyCacheHit,
        { retain: options.cacheRetention !== "transient" },
      )
    : await getOrLoadField(assemblyKey, () => loadAssembly(), onAssemblyCacheHit,
        { retain: options.cacheRetention !== "transient" });
  throwIfFieldReadAborted(options.signal);
  const provenance = externalFieldProvenance.get(assembled);
  if (provenance) externalProviderProvenance.set(makeFieldKey(runDir, stepIndex, fieldName, expectedGeneration), provenance);
  return assembled;
}

/** Rank-checked bounded-volume wrapper. */
export async function readFieldWindows(
  runDir: string,
  stepIndex: number,
  fieldName: string,
  expectedGeneration: string | null,
  windows: readonly FieldLoadWindow[],
  options: {
    latestScope?: string;
    signal?: AbortSignal;
    priority?: "interactive" | "background";
    brickHierarchyLevel?: number;
    cacheRetention?: "resident" | "transient";
  } = {},
): Promise<Npy3D> {
  const assembled = await readFieldAnyWindows(
    runDir,
    stepIndex,
    fieldName,
    expectedGeneration,
    windows,
    options,
  );
  if (assembled.shape.length !== 3) {
    throw new Error(`Bounded volume read returned rank ${assembled.shape.length}`);
  }
  return {
    shape: assembled.shape as [number, number, number],
    data: assembled.data,
    ...(assembled.windowGeometry ? { windowGeometry: assembled.windowGeometry } : {}),
  };
}

/** Scalar reduction of one field, computed on the Rust side (the whole array
 *  never crosses IPC). NaN where the field has no finite values. */
export interface FieldReduction {
  min: number;
  max: number;
  mean: number;
  rms: number;
  /** Count of finite elements / total elements — exposes a NaN-heavy field. */
  finite: number;
  total: number;
  /** Cells passing the requested threshold (0 when no threshold was given):
   *  countAbove = v > t, countBelow = v < t, countAbsAbove = |v| > t. Back the
   *  area-style time-series metrics (mesocyclone area, cold-pool extent). */
  countAbove: number;
  countBelow: number;
  countAbsAbove: number;
}

interface RawFieldReduction {
  min: number | null;
  max: number | null;
  mean: number | null;
  rms: number | null;
  finite: number;
  total: number;
  countAbove: number;
  countBelow: number;
  countAbsAbove: number;
}

interface RawAxisReduction {
  values: (number | null)[];
  axis: number;
  axisLen: number;
}

/** Max concurrent per-step reductions for time-series/time-height plots. Bounds transient Rust-pool memory (<=4 × field bytes). */
export const REDUCE_CONCURRENCY = 4;

/** Reduce a field to min/max/mean/rms on the Rust side; only a handful of numbers cross IPC. */
export async function reduceField(
  runDir: string,
  stepIndex: number,
  fieldName: string,
  expectedGeneration: string | null,
  threshold?: number,
): Promise<FieldReduction> {
  if (isPackagedAtmosphericExampleUri(runDir)) {
    return reducePackagedAtmosphericExampleField(
      runDir,
      stepIndex,
      fieldName,
      expectedGeneration,
      threshold,
    );
  }
  if (!isTauriRuntime()) {
    throw new Error("reduceField requires the Tauri runtime");
  }
  const r = await invokeCommand<RawFieldReduction>("reduce_field", {
    runDir,
    stepIndex,
    fieldName,
    threshold: threshold ?? null,
    expectedGeneration,
  });
  return {
    min: r.min ?? NaN,
    max: r.max ?? NaN,
    mean: r.mean ?? NaN,
    rms: r.rms ?? NaN,
    finite: r.finite,
    total: r.total,
    countAbove: r.countAbove,
    countBelow: r.countBelow,
    countAbsAbove: r.countAbsAbove,
  };
}

export type FieldReducer = "max" | "min" | "mean" | "rms";

/** Reduce a 3D field along one axis (0=x,1=y,2=z) Rust-side; returns one value per index (NaN for empty slabs). Backs the Hovmöller plot. */
export async function reduceFieldAxis(
  runDir: string,
  stepIndex: number,
  fieldName: string,
  axis: 0 | 1 | 2,
  reducer: FieldReducer,
  expectedGeneration: string | null,
): Promise<(number | null)[]> {
  if (isPackagedAtmosphericExampleUri(runDir)) {
    return reducePackagedAtmosphericExampleFieldAxis(
      runDir,
      stepIndex,
      fieldName,
      axis,
      reducer,
      expectedGeneration,
    );
  }
  if (!isTauriRuntime()) {
    throw new Error("reduceFieldAxis requires the Tauri runtime");
  }
  const r = await invokeCommand<RawAxisReduction>("reduce_field_axis", {
    runDir,
    stepIndex,
    fieldName,
    axis,
    reducer,
    expectedGeneration,
  });
  return r.values;
}

/** Column layout of trajectories.npy's last axis (matches the simulator's
 *  TrajectorySample): position + velocity + the vertical-vorticity budget
 *  (tilting/stretching/baroclinic — the tornadogenesis-mechanism terms, NaN on
 *  grids where the online budget is unsupported). */
export const TRAJECTORY_VARS = [
  "t",
  "x",
  "y",
  "z",
  "w",
  "zeta",
  "tilting",
  "stretching",
  "baroclinic",
] as const;

/** Read trajectories.npy: shape (n_parcels, n_samples, 9), NaN-padded. Columns are TRAJECTORY_VARS. Returns null outside Tauri or when absent. */
export async function readTrajectories(
  runDir: string,
  expectedGeneration: string | null,
): Promise<NpyN | null> {
  if (!isTauriRuntime()) return null;
  try {
    const buffer = await invokeCommand<ArrayBuffer>("read_trajectories", {
      runDir,
      expectedGeneration,
    });
    return parseNpy(buffer, { expectedRank: 3, expectedLastDimension: 9 });
  } catch {
    return null; // NotFound — this run has no trajectory output
  }
}

/**
 * Read the storm-track summary. Returns null in browser, when tracking is disabled, or mid-publication.
 * Callers must treat null as "not yet" — the file is rewritten continuously.
 */
export async function readStormTrack(
  runDir: string,
  expectedGeneration: string | null,
): Promise<StormTrackSnapshot | null> {
  if (!isTauriRuntime()) return null;
  const key = JSON.stringify([runDir, expectedGeneration]);
  const pending = stormTrackReads.get(key);
  if (pending) return pending;
  const read = (async () => {
    try {
      const contents = await invokeCommand<string | null>("read_storm_track", {
        runDir,
        expectedGeneration,
      });
      if (!contents) return null;
      const parsed = parseStormTrack(contents);
      return parsed.ok ? parsed.snapshot : null;
    } catch {
      return null;
    }
  })();
  stormTrackReads.set(key, read);
  void read.finally(() => {
    if (stormTrackReads.get(key) === read) stormTrackReads.delete(key);
  });
  return read;
}

/** Cheap existence probe for trajectories.npy. Avoids reading the multi-MB file. Returns false outside Tauri or on error. */
export async function runHasTrajectories(
  runDir: string,
  expectedGeneration: string | null,
): Promise<boolean> {
  if (isPackagedAtmosphericExampleUri(runDir)) return false;
  if (!isTauriRuntime()) return false;
  try {
    return await invokeCommand<boolean>("run_has_trajectories", {
      runDir,
      expectedGeneration,
    });
  } catch {
    return false;
  }
}

/* ---------- Internals ---------- */

interface RawStepInfo {
  index: number;
  simulationTimeS: number | null;
  fields: string[];
  grid: RawGridInfo | null;
  fieldUnits?: Record<string, string> | null;
  fieldCompression?: Record<string, FieldCompressionInfo> | null;
  fieldBrickHierarchies?: Record<string, FieldBrickHierarchyInfo> | null;
  publicationState?: StepPublicationState;
  publicationGeneration?: string | null;
  motion?: FrameMotion | null;
}

interface RawGridInfo {
  nr: number | null;
  nth: number | null;
  nz: number | null;
  coordinateSystem: string | null;
  staggering: string | null;
  drM: number | null;
  dyM: number | null;
  dthetaRad: number | null;
  dzM: number | null;
  terrainFollowing?: boolean;
  terrainGeometryIdentity?: string | null;
  horizontalScalarOriginM?: [number, number] | null;
  scalarOriginM: [number, number, number] | null;
  scalarCoordinatesM?: [number[], number[], number[]] | null;
  physicalBoundsM: AxisBounds | null;
  validIndexRange: IndexBounds | null;
  haloWidth: IndexBounds | null;
}

function normalizeStep(s: RawStepInfo): StepInfo {
  return {
    index: s.index,
    simulationTimeS: s.simulationTimeS,
    fields: s.fields,
    grid: s.grid ? normalizeGrid(s.grid) : null,
    fieldUnits: s.fieldUnits ?? {},
    fieldCompression: s.fieldCompression ?? {},
    fieldBrickHierarchies: s.fieldBrickHierarchies ?? {},
    fieldOrigins: Object.fromEntries(s.fields.map((field) => [field, "native" as const])),
    // Absent publicationState: older bridge → legacy (preserves pre-contract runs).
    publicationState: s.publicationState ?? "legacy",
    publicationGeneration: s.publicationGeneration ?? null,
    // Carried verbatim including unrecognised modes; frameMotion.ts is the gatekeeper.
    motion: s.motion ?? null,
  };
}

function normalizeGrid(g: RawGridInfo): GridInfo {
  return {
    nr: g.nr ?? undefined,
    nth: g.nth ?? undefined,
    nz: g.nz ?? undefined,
    coordinateSystem: g.coordinateSystem ?? undefined,
    staggering: g.staggering ?? undefined,
    drM: g.drM ?? undefined,
    dyM: g.dyM ?? undefined,
    dthetaRad: g.dthetaRad ?? undefined,
    dzM: g.dzM ?? undefined,
    terrainFollowing: g.terrainFollowing ?? false,
    terrainGeometryIdentity: g.terrainGeometryIdentity ?? undefined,
    horizontalScalarOriginM: g.horizontalScalarOriginM ?? undefined,
    scalarOriginM: g.scalarOriginM ?? undefined,
    scalarCoordinatesM: g.scalarCoordinatesM ?? undefined,
    physicalBoundsM: g.physicalBoundsM ?? undefined,
    validIndexRange: g.validIndexRange ?? undefined,
    haloWidth: g.haloWidth ?? undefined,
  };
}
