/** Cross-cutting shared types for Nimbus Studio. */
import type { RemoteComputeTarget, RemoteReceiveSummary } from "@/lib/platform/remoteCompute";

export type PageId = "projects" | "config" | "runs" | "plots" | "settings";

export type SchemeId =
  | "klemp_wilhelmson_split_cartesian_cgrid"
  | "klemp_wilhelmson_split_cartesian_agrid"
  | "cartesian"
  | "cart"
  | "cartesian_cpu";

export type RunStatus = "queued" | "running" | "completed" | "failed" | "aborted";

export interface RunProgress {
  /** Latest simulated time (s) parsed from stdout. */
  simTimeS: number;
  /** Latest step counter, if available. */
  step: number | null;
  /** Latest max |velocity| value, parsed from [VEL-MAX] lines. */
  velMax: number | null;
}

export interface QueuedRun {
  id: string;
  caseName: string;
  scheme: SchemeId;
  status: RunStatus;
  executionTarget?: RemoteComputeTarget | null;
  remoteTerminationUnconfirmed?: boolean;
  remoteTransferInProgress?: boolean;
  remoteReceive?: RemoteReceiveSummary | null;
  /** Unix ms timestamp the simulator was spawned. Null for queued/mock
   *  runs that haven't started. The RunDetail wall clock derives from
   *  this plus a local 1Hz tick — keeps the counter smooth without
   *  paying the cost of a 1s IPC poll cycle. */
  startedAtMs: number | null;
  /** Unix ms timestamp the child exited. Null while running; set on
   *  completed/failed/aborted. Freezes the wall clock on the run-detail
   *  view instead of letting (now() - startedAtMs) tick forever. */
  endedAtMs: number | null;
  /** Last reported simulated time. */
  simTimeS: number | null;
  /** Latest step counter parsed from progress lines, if available. */
  step: number | null;
  /** Latest max |velocity| from [VEL-MAX] lines, if available. */
  velMax: number | null;
  /** Configured duration_s, for progress estimate. */
  durationS: number;
  /** Absolute path the simulator writes step_NNNNNN/ subdirs into.
   *  Null if the run was tracked from a foreign source. */
  outputDir: string | null;
  /** Whether the row came from the live process registry or was rebuilt by
   *  scanning the configured runs directory after Studio startup. */
  origin?: "live" | "disk";
  /** Read-only disk qualification. Partial/invalid runs remain visible so a
   *  scientist can diagnose or reveal them without Studio inventing data. */
  libraryHealth?: "ready" | "partial" | "invalid";
  libraryIssue?: string | null;
  frameCount?: number;
  fieldCount?: number;
  outputFormat?: string;
  modifiedAtMs?: number | null;
  libraryScanLimited?: boolean;
  libraryScanLimit?: number;
  provenance?: RunProvenanceSummary;
}

export interface RunProvenanceSummary {
  health: "verified" | "missing" | "invalid";
  issue: string | null;
  runtimeHealth: "verified" | "missing" | "invalid";
  runtimeIssue: string | null;
  runId: string | null;
  createdUnixMs: number | null;
  executablePath: string | null;
  executableDigest: string | null;
  configurationDigest: string | null;
  catalogFingerprint: string | null;
  requestedBackend: string | null;
  requestedFallbackAllowed: boolean | null;
  requestedOutputFormat: string | null;
  requestedExecutionMode: string | null;
  requestedValidationMode: string | null;
  requestedSchemes: Record<string, string>;
  soundingRequestedPath: string | null;
  soundingCanonicalPath: string | null;
  soundingDigest: string | null;
  soundingIssue: string | null;
  runtimeActualsAvailable: boolean;
  actualBackend: string | null;
  fallbackActive: boolean | null;
  actualSchemes: Record<string, string>;
  finalStatus: string | null;
  exitCode: number | null;
  processStartedUnixMs: number | null;
  processEndedUnixMs: number | null;
  completeFrameCount: number | null;
  actualOutputFormat: string | null;
  verifiedCheckpointCount: number | null;
  latestCheckpointPath: string | null;
}

export interface RunLibraryEntry {
  executionTarget?: RemoteComputeTarget | null;
  remoteReceive?: RemoteReceiveSummary | null;
  id: string;
  caseName: string;
  outputDir: string;
  health: "ready" | "partial" | "invalid";
  issue: string | null;
  startedAtMs: number | null;
  modifiedAtMs: number | null;
  frameCount: number;
  latestStep: number | null;
  simulationTimeS: number | null;
  fieldCount: number;
  outputFormat: string;
  scanLimited: boolean;
  scanLimit: number;
  provenance: RunProvenanceSummary;
}

/* ---------- Tauri event payloads (match src-tauri/src/commands/runner.rs) ---------- */

export interface LineEvent {
  runId: string;
  stream: "stdout" | "stderr";
  line: string;
  ts: number;
}

export interface ProgressEvent {
  runId: string;
  progress: RunProgress;
}

export interface ExitEvent {
  runId: string;
  status: RunStatus;
  exitCode: number | null;
}

/* ---------- RunInfo (from list_runs / poll_run) ---------- */

export interface RunInfo {
  id: string;
  caseName: string;
  status: RunStatus;
  executionTarget?: RemoteComputeTarget | null;
  remoteTerminationUnconfirmed?: boolean;
  remoteTransferInProgress?: boolean;
  remoteReceive?: RemoteReceiveSummary | null;
  pid: number;
  startedAtMs: number;
  endedAtMs: number | null;
  progress: RunProgress | null;
  outputDir: string;
  /** Dynamics scheme captured at spawn from the config (e.g.
   *  "klemp_wilhelmson_split_cartesian_cgrid"). Empty string when the run
   *  predates this field. */
  scheme: string;
  /** Configured run length in seconds (v1 `time.run_time`). 0 when unknown. */
  durationS: number;
}

export interface SimulatorStatus {
  binaryPath: string | null;
  /** "ready" if `bin/nimbus_sim` resolves; otherwise the missing-path reason. */
  state: "ready" | "missing" | "unknown";
  /** GPU vendor / device label if Vulkan is available, else null. */
  gpu: string | null;
}

export interface HardwareInfo {
  cpuBrand: string | null;
  coreCount: number | null;
  ramBytes: number | null;
}

export interface ProcessMemoryInfo {
  schema: "nimbus.studio.process-memory";
  schemaVersion: 3;
  currentRssBytes: number | null;
  peakRssBytes: number | null;
  processTreeRssBytes: number | null;
  processTreePeakRssBytes: number | null;
  processTreeProcessCount: number | null;
  processTreeCoverage: "whole_application" | "host_plus_descendants" | "host_only";
  measurement: string;
}
