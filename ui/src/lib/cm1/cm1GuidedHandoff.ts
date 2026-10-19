import { isLocalDatasetSource } from "@/lib/comparison/comparisonSession";
import {
  createLocalDatasetSource,
  datasetSourcePaths,
  type LocalDatasetSource,
} from "@/lib/dataset/datasetSources";

export const CM1_GUIDED_HANDOFF_SCHEMA = "nimbus.studio.cm1-guided-handoff" as const;
export const CM1_GUIDED_HANDOFF_VERSION = 2 as const;
export const CM1_GUIDED_HANDOFF_STORAGE_KEY = "nimbus:cm1-guided-handoff:v2";
export const CM1_SMOKE_DURATION_S = 30 as const;
export const CM1_SMOKE_OUTPUT_INTERVAL_S = 10 as const;
const MAX_SOURCE_PATHS = 512;
const MAX_PATH_CHARS = 4096;
const MAX_YAML_CHARS = 512_000;
const MAX_BLOCKERS_STORED = 12;
const MAX_BLOCKER_CHARS = 512;

export type Cm1HandoffRun =
  | { status: "not_started"; runId: null; outputDir: null; pendingReason: null }
  | { status: "running" | "pending"; runId: string; outputDir: string | null; pendingReason: string }
  | { status: "complete"; runId: string; outputDir: string; pendingReason: null }
  | { status: "failed"; runId: string; outputDir: string | null; pendingReason: string };

export interface Cm1GuidedHandoff {
  schema: typeof CM1_GUIDED_HANDOFF_SCHEMA;
  schemaVersion: typeof CM1_GUIDED_HANDOFF_VERSION;
  id: string;
  updatedAt: string;
  cm1Source: LocalDatasetSource;
  evidence: {
    historyFileCount: number;
    soundingSourcePath: string;
    projectedLevelCount: number;
    /** Optional in schema v2 so handoffs saved before native CM1 support remain readable. */
    soundingSourceMode?: "reviewed_projection" | "native_cm1";
    mappingCounts: {
      ready: number;
      needsDecision: number;
      different: number;
      unsupported: number;
    };
    mappingBlockers: string[];
    mappingBlockerTotal: number;
    mappingBlockersTruncated: boolean;
    reviewedAt: string;
  };
  smoke: {
    soundingPath: string;
    configPath: string;
    yaml: string;
    configurationDigest: string;
    durationS: typeof CM1_SMOKE_DURATION_S;
    outputIntervalS: typeof CM1_SMOKE_OUTPUT_INTERVAL_S;
    catalogFingerprint: string;
    executablePath: string;
    initializationExecutablePath: string;
    stagedAt: string;
    consumedAt: string | null;
  };
  run: Cm1HandoffRun;
  comparisonOpenedAt: string | null;
}

export interface CreateCm1GuidedHandoffInput {
  cm1Source: LocalDatasetSource;
  soundingSourcePath: string;
  projectedLevelCount: number;
  soundingSourceMode?: "reviewed_projection" | "native_cm1";
  mappingCounts: Cm1GuidedHandoff["evidence"]["mappingCounts"];
  mappingBlockers: readonly string[];
  soundingPath: string;
  configPath: string;
  yaml: string;
  catalogFingerprint: string;
  executablePath: string;
  initializationExecutablePath: string;
  now?: string;
}

type StorageReader = Pick<Storage, "getItem">;
type StorageWriter = Pick<Storage, "setItem">;

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function isBoundedString(value: unknown, maxChars: number): value is string {
  return isNonEmptyString(value) && value.length <= maxChars;
}

function isCount(value: unknown): value is number {
  return Number.isInteger(value) && (value as number) >= 0;
}

export function fnv1a64Text(value: string): string {
  let hash = 0xcbf29ce484222325n;
  for (const byte of new TextEncoder().encode(value)) {
    hash ^= BigInt(byte);
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return `fnv1a64:${hash.toString(16).padStart(16, "0")}`;
}

function compactCm1Source(source: LocalDatasetSource): LocalDatasetSource {
  const historyPaths = source.caseInventory?.historyFiles;
  return createLocalDatasetSource(
    "cm1_netcdf",
    historyPaths && historyPaths.length > 0 ? historyPaths : datasetSourcePaths(source),
  );
}

export function createCm1GuidedHandoff(
  input: CreateCm1GuidedHandoffInput,
): Cm1GuidedHandoff {
  if (input.cm1Source.kind !== "cm1_netcdf") {
    throw new Error("A guided CM1 handoff requires a CM1 / netCDF source.");
  }
  const paths = input.cm1Source.caseInventory?.historyFiles.length
    ? input.cm1Source.caseInventory.historyFiles
    : datasetSourcePaths(input.cm1Source);
  const historyFileCount = paths.length;
  if (historyFileCount < 1 || paths.length < 1) {
    throw new Error("A guided CM1 handoff requires at least one CM1 history file.");
  }
  if (paths.length > MAX_SOURCE_PATHS || paths.some((path) => !isBoundedString(path, MAX_PATH_CHARS))) {
    throw new Error(`The guided handoff supports at most ${MAX_SOURCE_PATHS} bounded local history paths.`);
  }
  if (input.projectedLevelCount < 1) {
    throw new Error("A reviewed CM1 sounding projection is required.");
  }
  const requiredPaths = [
    input.soundingSourcePath,
    input.soundingPath,
    input.configPath,
    input.executablePath,
    input.initializationExecutablePath,
  ];
  if (requiredPaths.some((value) => !isBoundedString(value, MAX_PATH_CHARS))
    || !isBoundedString(input.yaml, MAX_YAML_CHARS)
    || !isBoundedString(input.catalogFingerprint, 512)) {
    throw new Error("The reviewed sounding and exact-binary evidence must be complete.");
  }
  if (input.mappingBlockers.some((value) => !isBoundedString(value, MAX_BLOCKER_CHARS))) {
    throw new Error("CM1 mapping blocker evidence exceeds the bounded workflow record.");
  }
  const now = input.now ?? new Date().toISOString();
  return {
    schema: CM1_GUIDED_HANDOFF_SCHEMA,
    schemaVersion: CM1_GUIDED_HANDOFF_VERSION,
    id: `cm1-handoff-${now}`,
    updatedAt: now,
    cm1Source: compactCm1Source(input.cm1Source),
    evidence: {
      historyFileCount,
      soundingSourcePath: input.soundingSourcePath,
      projectedLevelCount: input.projectedLevelCount,
      soundingSourceMode: input.soundingSourceMode ?? "reviewed_projection",
      mappingCounts: { ...input.mappingCounts },
      mappingBlockers: input.mappingBlockers.slice(0, MAX_BLOCKERS_STORED),
      mappingBlockerTotal: input.mappingBlockers.length,
      mappingBlockersTruncated: input.mappingBlockers.length > MAX_BLOCKERS_STORED,
      reviewedAt: now,
    },
    smoke: {
      soundingPath: input.soundingPath,
      configPath: input.configPath,
      yaml: input.yaml,
      configurationDigest: fnv1a64Text(input.yaml),
      durationS: CM1_SMOKE_DURATION_S,
      outputIntervalS: CM1_SMOKE_OUTPUT_INTERVAL_S,
      catalogFingerprint: input.catalogFingerprint,
      executablePath: input.executablePath,
      initializationExecutablePath: input.initializationExecutablePath,
      stagedAt: now,
      consumedAt: null,
    },
    run: { status: "not_started", runId: null, outputDir: null, pendingReason: null },
    comparisonOpenedAt: null,
  };
}

export function parseCm1GuidedHandoff(value: unknown): Cm1GuidedHandoff | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Partial<Cm1GuidedHandoff>;
  const evidence = record.evidence as Partial<Cm1GuidedHandoff["evidence"]> | undefined;
  const counts = evidence?.mappingCounts as Partial<Cm1GuidedHandoff["evidence"]["mappingCounts"]> | undefined;
  const smoke = record.smoke as Partial<Cm1GuidedHandoff["smoke"]> | undefined;
  const run = record.run as Partial<Cm1HandoffRun> | undefined;
  const runValid = run?.status === "not_started"
    ? run.runId === null && run.outputDir === null && run.pendingReason === null
    : run?.status === "running" || run?.status === "pending" || run?.status === "failed"
      ? isNonEmptyString(run.runId)
        && (run.outputDir === null || isBoundedString(run.outputDir, MAX_PATH_CHARS))
        && isBoundedString(run.pendingReason, 1_000)
      : run?.status === "complete"
        ? isNonEmptyString(run.runId)
          && isBoundedString(run.outputDir, MAX_PATH_CHARS)
          && run.pendingReason === null
        : false;
  return record.schema === CM1_GUIDED_HANDOFF_SCHEMA
    && record.schemaVersion === CM1_GUIDED_HANDOFF_VERSION
    && isBoundedString(record.id, 512)
    && isBoundedString(record.updatedAt, 64)
    && isLocalDatasetSource(record.cm1Source)
    && record.cm1Source.kind === "cm1_netcdf"
    && isCount(evidence?.historyFileCount) && evidence.historyFileCount > 0
    && datasetSourcePaths(record.cm1Source).length <= MAX_SOURCE_PATHS
    && datasetSourcePaths(record.cm1Source).every((path) => isBoundedString(path, MAX_PATH_CHARS))
    && isBoundedString(evidence.soundingSourcePath, MAX_PATH_CHARS)
    && isCount(evidence.projectedLevelCount) && evidence.projectedLevelCount > 0
    && (evidence.soundingSourceMode === undefined
      || evidence.soundingSourceMode === "reviewed_projection"
      || evidence.soundingSourceMode === "native_cm1")
    && isCount(counts?.ready)
    && isCount(counts.needsDecision)
    && isCount(counts.different)
    && isCount(counts.unsupported)
    && Array.isArray(evidence.mappingBlockers)
    && evidence.mappingBlockers.length <= MAX_BLOCKERS_STORED
    && evidence.mappingBlockers.every((value) => isBoundedString(value, MAX_BLOCKER_CHARS))
    && isCount(evidence.mappingBlockerTotal)
    && evidence.mappingBlockerTotal >= evidence.mappingBlockers.length
    && evidence.mappingBlockersTruncated === (evidence.mappingBlockerTotal > evidence.mappingBlockers.length)
    && isBoundedString(evidence.reviewedAt, 64)
    && isBoundedString(smoke?.soundingPath, MAX_PATH_CHARS)
    && isBoundedString(smoke.configPath, MAX_PATH_CHARS)
    && isBoundedString(smoke.yaml, MAX_YAML_CHARS)
    && /^fnv1a64:[a-f0-9]{16}$/.test(smoke.configurationDigest ?? "")
    && smoke.configurationDigest === fnv1a64Text(smoke.yaml)
    && smoke.durationS === CM1_SMOKE_DURATION_S
    && smoke.outputIntervalS === CM1_SMOKE_OUTPUT_INTERVAL_S
    && isBoundedString(smoke.catalogFingerprint, 512)
    && isBoundedString(smoke.executablePath, MAX_PATH_CHARS)
    && isBoundedString(smoke.initializationExecutablePath, MAX_PATH_CHARS)
    && isBoundedString(smoke.stagedAt, 64)
    && (smoke.consumedAt === null || isBoundedString(smoke.consumedAt, 64))
    && runValid
    && (record.comparisonOpenedAt === null || isBoundedString(record.comparisonOpenedAt, 64))
    ? record as Cm1GuidedHandoff
    : null;
}

export function loadCm1GuidedHandoff(storage: StorageReader): Cm1GuidedHandoff | null {
  try {
    const raw = storage.getItem(CM1_GUIDED_HANDOFF_STORAGE_KEY);
    return raw ? parseCm1GuidedHandoff(JSON.parse(raw)) : null;
  } catch {
    return null;
  }
}

export function saveCm1GuidedHandoff(
  storage: StorageWriter,
  record: Cm1GuidedHandoff,
): void {
  const parsed = parseCm1GuidedHandoff(record);
  if (!parsed) throw new Error("Refusing to save incomplete CM1 handoff evidence.");
  storage.setItem(CM1_GUIDED_HANDOFF_STORAGE_KEY, JSON.stringify(parsed));
}

export function consumeCm1SmokeConfig(record: Cm1GuidedHandoff, now = new Date().toISOString()): Cm1GuidedHandoff {
  return { ...record, updatedAt: now, smoke: { ...record.smoke, consumedAt: now } };
}

export function cm1StagedLaunchMismatch(
  record: Cm1GuidedHandoff,
  evidence: { yaml: string; catalogFingerprint: string; executablePath: string },
): string | null {
  if (evidence.yaml !== record.smoke.yaml || fnv1a64Text(evidence.yaml) !== record.smoke.configurationDigest) {
    return "Exact staged YAML digest changed.";
  }
  if (evidence.catalogFingerprint !== record.smoke.catalogFingerprint) {
    return "Exact simulator catalog fingerprint changed.";
  }
  if (evidence.executablePath !== record.smoke.executablePath) {
    return "Exact simulator executable identity changed.";
  }
  return null;
}

export function markCm1SmokeRunning(
  record: Cm1GuidedHandoff,
  runId: string,
  outputDir: string | null,
  now = new Date().toISOString(),
): Cm1GuidedHandoff {
  if (!isNonEmptyString(runId) || record.smoke.consumedAt === null) {
    throw new Error("Consume the staged smoke configuration before recording a run.");
  }
  const canonicalOutput = outputDir?.trim() || null;
  return {
    ...record,
    updatedAt: now,
    run: {
      status: "running",
      runId,
      outputDir: canonicalOutput,
      pendingReason: canonicalOutput
        ? "Waiting for durable Run Library completion evidence."
        : "Launch started, but its output directory identity is unavailable; comparison remains blocked.",
    },
  };
}

export function markCm1SmokeFailed(
  record: Cm1GuidedHandoff,
  reason: string,
  now = new Date().toISOString(),
): Cm1GuidedHandoff {
  if (record.run.status !== "running" && record.run.status !== "pending" && record.run.status !== "complete") {
    throw new Error("No guided smoke run is active.");
  }
  if (!isBoundedString(reason, 1_000)) throw new Error("A bounded failure reason is required.");
  return {
    ...record,
    updatedAt: now,
    run: { status: "failed", runId: record.run.runId, outputDir: record.run.outputDir, pendingReason: reason },
  };
}

export function markCm1SmokePending(
  record: Cm1GuidedHandoff,
  reason: string,
  now = new Date().toISOString(),
): Cm1GuidedHandoff {
  if (record.run.status !== "running" && record.run.status !== "pending" && record.run.status !== "complete") {
    throw new Error("No guided smoke run is active.");
  }
  if (!isBoundedString(reason, 1_000)) throw new Error("A bounded pending reason is required.");
  return {
    ...record,
    updatedAt: now,
    run: { ...record.run, status: "pending", pendingReason: reason },
  };
}

export interface Cm1DurableRunEvidence {
  outputDir: string;
  health: "ready" | "partial" | "invalid";
  issue: string | null;
  frameCount: number;
  provenance: {
    health: "verified" | "missing" | "invalid";
    runtimeHealth: "verified" | "missing" | "invalid";
    runId: string | null;
    executablePath: string | null;
    configurationDigest: string | null;
    catalogFingerprint: string | null;
    finalStatus: string | null;
    exitCode: number | null;
    completeFrameCount: number | null;
  };
}

export function reconcileCm1SmokeRun(
  record: Cm1GuidedHandoff,
  evidence: Cm1DurableRunEvidence | null,
  now = new Date().toISOString(),
): Cm1GuidedHandoff {
  if (record.run.status !== "running" && record.run.status !== "pending" && record.run.status !== "complete") {
    return record;
  }
  const pending = (reason: string): Cm1GuidedHandoff => markCm1SmokePending(record, reason, now);
  if (!evidence) return pending("Durable Run Library scan has not found the matching output directory yet.");
  if (record.run.outputDir
    && evidence.outputDir.replace(/[\\/]+$/, "") !== record.run.outputDir.replace(/[\\/]+$/, "")) {
    return pending("Durable output directory does not match the exact launch identity.");
  }
  if (evidence.provenance.runId !== record.run.runId) return pending("Durable provenance run identity does not match the guided launch.");
  if (evidence.provenance.health !== "verified") return pending("Launch provenance is missing or invalid.");
  if (evidence.provenance.runtimeHealth !== "verified") return pending("Completion provenance is missing or invalid.");
  if (evidence.provenance.configurationDigest !== record.smoke.configurationDigest) return pending("Durable YAML digest does not match the exact staged smoke configuration.");
  if (evidence.provenance.catalogFingerprint !== record.smoke.catalogFingerprint) return pending("Durable catalog fingerprint does not match the staged executable evidence.");
  if (evidence.provenance.executablePath !== record.smoke.executablePath) return pending("Durable executable identity does not match the staged executable.");
  if (evidence.provenance.finalStatus === "failed" || evidence.provenance.finalStatus === "aborted") {
    return markCm1SmokeFailed(record, `Runtime completion provenance reports ${evidence.provenance.finalStatus}.`, now);
  }
  if (evidence.provenance.finalStatus !== "completed" || evidence.provenance.exitCode !== 0) return pending("Completion provenance does not report a successful exit.");
  if (evidence.health !== "ready") return pending(evidence.issue ?? "Run Library output health is not ready.");
  if (evidence.frameCount <= 0 || (evidence.provenance.completeFrameCount ?? 0) <= 0) {
    return pending("No complete output frames are durably qualified for comparison.");
  }
  return {
    ...record,
    updatedAt: now,
    run: { status: "complete", runId: record.run.runId, outputDir: evidence.outputDir, pendingReason: null },
  };
}

export function cm1HandoffComparisonPair(record: Cm1GuidedHandoff): {
  left: LocalDatasetSource;
  right: LocalDatasetSource;
} | null {
  if (record.run.status !== "complete") return null;
  return {
    left: record.cm1Source,
    right: createLocalDatasetSource("nimbus_run", record.run.outputDir),
  };
}

export function markCm1ComparisonOpened(record: Cm1GuidedHandoff, now = new Date().toISOString()): Cm1GuidedHandoff {
  if (!cm1HandoffComparisonPair(record)) throw new Error("Comparison evidence is not ready.");
  return { ...record, updatedAt: now, comparisonOpenedAt: now };
}
