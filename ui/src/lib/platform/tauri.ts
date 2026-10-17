/*
 * Tauri bridge for Nimbus Studio.
 * Desktop: calls Tauri commands in src-tauri/.
 * Browser (no tauri:dev): falls back to mock data for design iteration.
 */

import type {
  ExitEvent,
  HardwareInfo,
  LineEvent,
  ProgressEvent,
  ProcessMemoryInfo,
  QueuedRun,
  RunInfo,
  RunLibraryEntry,
  SimulatorStatus,
} from "@/lib/core/types";
import { invokeCommand } from "@/lib/platform/appInvoke";
import { mockHardware, mockQueue, mockSimulator } from "@/lib/core/mockData";
import {
  MOCK_SCHEME_CATALOG,
  validateSchemeCatalog,
  type SchemeCatalog,
} from "@/lib/config/schemeCatalog";
import type { Cm1SoundingReport } from "@/lib/dataset/datasetSources";
import type { RemoteComputeTarget, RemoteComputeProbe, RemoteReceiveSummary } from "./remoteCompute";

interface TauriWindow {
  __TAURI_INTERNALS__?: unknown;
}

export function isTauriRuntime(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in (window as TauriWindow);
}

async function invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  if (!isTauriRuntime()) {
    throw new Error(`tauri::${cmd} called outside Tauri runtime`);
  }
  return invokeCommand<T>(cmd, args);
}

/* ---------- Commands ---------- */

const BROWSER_CONFIG_PREFIX = "nimbus.browser-config:";

function browserConfigKey(path: string): string {
  return `${BROWSER_CONFIG_PREFIX}${path}`;
}

export async function readConfig(path: string): Promise<string> {
  if (!isTauriRuntime()) {
    try {
      const saved = window.sessionStorage.getItem(browserConfigKey(path));
      if (saved !== null) return saved;
    } catch {
      // Storage is optional in sandboxed browser previews.
    }
    return `# mock: ${path}\ncase_name: cgrid_tornado_genesis\nduration_s: 1200\n`;
  }
  return invoke<string>("read_config", { path });
}

export async function writeConfig(path: string, contents: string): Promise<void> {
  if (!isTauriRuntime()) {
    try {
      window.sessionStorage.setItem(browserConfigKey(path), contents);
    } catch (reason) {
      throw new Error(
        `Browser preview could not retain '${path}' in session storage: ${reason instanceof Error ? reason.message : String(reason)}`,
      );
    }
    console.info(`[mock] writeConfig(${path}, ${contents.length} bytes)`);
    return;
  }
  return invoke<void>("write_config", { path, contents });
}

export interface ConfigMigrationResult {
  migratedYaml: string;
  stderr: string;
  exitCode: number | null;
}

export async function migrateConfig(
  configPath: string,
  simulatorPath?: string,
  executionTarget?: RemoteComputeTarget,
): Promise<ConfigMigrationResult> {
  if (!isTauriRuntime()) {
    throw new Error("Configuration migration requires Nimbus Studio desktop.");
  }
  return invoke<ConfigMigrationResult>("migrate_config", { configPath, simulatorPath, executionTarget });
}

export interface ExactConfigValidationResult {
  valid: boolean;
  stdout: string;
  stderr: string;
  exitCode: number | null;
}

export async function validateConfigContents(
  contents: string,
  configPath: string | null,
  simulatorPath?: string,
  executionTarget?: RemoteComputeTarget,
): Promise<ExactConfigValidationResult> {
  if (!isTauriRuntime()) {
    throw new Error("Exact configuration validation requires Nimbus Studio desktop.");
  }
  return invoke<ExactConfigValidationResult>("validate_config_contents", {
    contents,
    configPath,
    simulatorPath,
    executionTarget,
  });
}

/** Absolute path to the Nimbus project root. Anchors config paths away from ui/src-tauri/ (would trip dev watcher). */
export async function getProjectRoot(): Promise<string> {
  if (!isTauriRuntime()) {
    // Browser-mode mock: assume the dev shell is rooted at the repo.
    return "/__mock_project_root__";
  }
  return invoke<string>("project_root");
}

/** Writable, application-owned directory for cloned templates and new configs. */
export async function getUserConfigDirectory(): Promise<string> {
  if (!isTauriRuntime()) return "/__mock_user_config__";
  return invoke<string>("user_config_dir");
}

export async function listConfigs(directory: string): Promise<string[]> {
  if (!isTauriRuntime()) {
    const microphysics = ["dry", "kessler", "lin", "milbrandt", "thompson"];
    const profiles = [
      "baseline",
      "radiation",
      "ysu",
      "ysu_radiation",
      "les",
      "les_radiation",
      "tke",
      "tke_radiation",
    ];
    return microphysics.flatMap((scheme) =>
      profiles.map((profile) =>
        `1000meterResolution/${scheme}/${profile}.yaml`),
    );
  }
  return invoke<string[]>("list_configs", { directory });
}

/**
 * Discover every YAML file recursively under the public configs directory.
 * Results are folder-relative paths (for example,
 * "500meterResolution/kessler/ysu.yaml"). There is intentionally no folder
 * allowlist: user-created YAML files and future catalog subfolders appear in
 * the Studio picker automatically.
 */
export async function listTemplateConfigs(projectRoot?: string): Promise<string[]> {
  const base = projectRoot ? `${projectRoot}/configs` : "configs";
  return listConfigs(base);
}

export interface SpawnSimOptions {
  executionTarget?: RemoteComputeTarget;
  expectedExecutionTargetFingerprint?: string;
  /** Absolute path (or `~/...`) to a nimbus_sim binary that overrides
   *  PATH / build-tree auto-resolution. Backed by the Settings page. */
  binaryPath?: string;
  /** Exact catalog fingerprint returned by configuration preflight. Desktop
   *  re-queries the selected binary immediately before spawn and rejects a
   *  mismatch, closing the preflight-to-launch replacement gap. */
  expectedCatalogFingerprint?: string;
  /** Fingerprint of the exact executable bytes that passed preflight. */
  expectedBinaryFingerprint?: string;
  /** Fingerprint of the exact YAML bytes that passed validate-only. */
  expectedConfigFingerprint?: string;
  /** Root directory where the timestamped per-run subdir is created.
   *  When omitted the backend falls back to `$HOME/.nimbus/runs`.
   *  Mutually exclusive with `outputDir`. */
  runsRoot?: string;
  /** Exact output directory for this run, overriding the
   *  `runsRoot/timestamped-subdir` convention. Used by callers that
   *  pre-compute the full path (rare). */
  outputDir?: string;
  /** Dynamics scheme (e.g. "klemp_wilhelmson_split_cartesian_cgrid") from
   *  the config, stored on the run record so the queue + run-detail show the
   *  real scheme rather than a placeholder. Optional -- backend defaults when
   *  omitted. */
  scheme?: string;
  /** Configured run length in seconds (v1 `time.run_time`). Drives the
   *  progress-bar denominator. Optional — backend defaults when omitted. */
  durationS?: number;
  /** Values observed from the exact config/preflight request. Missing values
   * are serialized as unavailable in immutable provenance, never guessed. */
  requestedValues?: {
    backend?: string;
    allowFallback?: boolean;
    outputFormat?: string;
    executionMode?: string;
    validationMode?: string;
    soundingPath?: string;
    schemes: Record<string, string>;
  };
}

export async function spawnSim(
  configPath: string,
  opts?: SpawnSimOptions,
): Promise<string> {
  if (opts?.executionTarget && !isTauriRuntime()) throw new Error("Remote compute requires Nimbus Studio desktop.");
  if (!isTauriRuntime()) {
    console.info(`[mock] spawnSim(${configPath}, ${JSON.stringify(opts ?? {})})`);
    return `mock-run-${Date.now()}`;
  }
  return invoke<string>("spawn_sim", {
    configPath,
    executionTarget: opts?.executionTarget,
    expectedExecutionTargetFingerprint: opts?.expectedExecutionTargetFingerprint,
    simulatorPath: opts?.binaryPath,
    expectedCatalogFingerprint: opts?.expectedCatalogFingerprint,
    expectedBinaryFingerprint: opts?.expectedBinaryFingerprint,
    expectedConfigFingerprint: opts?.expectedConfigFingerprint,
    runsRoot: opts?.runsRoot,
    outputDir: opts?.outputDir,
    scheme: opts?.scheme,
    durationS: opts?.durationS,
    requestedValues: opts?.requestedValues,
  });
}

export async function abortSim(runId: string): Promise<void> {
  if (!isTauriRuntime()) {
    console.info(`[mock] abortSim(${runId})`);
    return;
  }
  return invoke<void>("abort_sim", { runId });
}

export async function receiveRemoteResults(runDir: string): Promise<RemoteReceiveSummary> {
  return invoke<RemoteReceiveSummary>("receive_remote_results", { runDir });
}

export async function cancelRemoteReceive(runDir: string): Promise<void> {
  return invoke<void>("cancel_remote_receive", { runDir });
}

/** Drop a run from the registry (completed/failed/aborted only). Output files are not touched. */
export async function deleteRun(runId: string): Promise<void> {
  if (!isTauriRuntime()) {
    console.info(`[mock] deleteRun(${runId})`);
    return;
  }
  return invoke<void>("delete_run", { runId });
}

/** Reveal a path in the OS file manager. Uses a custom Rust command (plugin-shell's scope rejects fs paths). */
export async function openPath(path: string): Promise<void> {
  if (!isTauriRuntime()) {
    console.info(`[mock] openPath(${path})`);
    return;
  }
  return invoke<void>("reveal_path", { path });
}

export async function listRuns(): Promise<QueuedRun[]> {
  if (!isTauriRuntime()) return mockQueue;
  const raw = await invoke<RunInfo[]>("list_runs");
  // Map RunInfo (Rust) → QueuedRun (UI). Fall back to Cartesian C-grid id for old run records.
  return raw.map((r) => ({
    id: r.id,
    caseName: r.caseName,
    scheme: (r.scheme || "klemp_wilhelmson_split_cartesian_cgrid") as QueuedRun["scheme"],
    status: r.status,
    executionTarget: r.executionTarget,
    remoteTerminationUnconfirmed: r.remoteTerminationUnconfirmed,
    remoteTransferInProgress: r.remoteTransferInProgress,
    remoteReceive: r.remoteReceive,
    startedAtMs: r.startedAtMs > 0 ? r.startedAtMs : null,
    endedAtMs: r.endedAtMs,
    simTimeS: r.progress?.simTimeS ?? null,
    step: r.progress?.step ?? null,
    velMax: r.progress?.velMax ?? null,
    durationS: r.durationS > 0 ? r.durationS : 1200,
    outputDir: r.outputDir || null,
  }));
}

/** Rebuild the read-only local run index from the configured runs root. */
export async function discoverRunLibrary(runsRoot: string): Promise<RunLibraryEntry[]> {
  if (!isTauriRuntime()) return [];
  return invoke<RunLibraryEntry[]>("discover_run_library", { runsRoot });
}

/** Export validated launch + runtime provenance as one read-only JSON bundle. */
export async function copyRunProvenance(
  runDir: string,
  destination: string,
): Promise<string> {
  if (!isTauriRuntime()) return destination;
  return invoke<string>("copy_run_provenance", { runDir, destination });
}

export async function pickProvenanceSavePath(
  defaultName = "nimbus-run-evidence.json",
): Promise<string | null> {
  if (!isTauriRuntime()) return null;
  const { save } = await import("@tauri-apps/plugin-dialog");
  return save({
    title: "Export run evidence bundle",
    defaultPath: defaultName,
    filters: [{ name: "JSON evidence bundle", extensions: ["json"] }],
  });
}

export async function getSimulatorStatus(binaryPath?: string): Promise<SimulatorStatus> {
  if (!isTauriRuntime()) return mockSimulator;
  return invoke<SimulatorStatus>("simulator_status", {
    simulatorPath: binaryPath,
  });
}

/** Exact selectable-scheme metadata from the configured simulator binary. */
export async function getSchemeCatalog(binaryPath?: string, executionTarget?: RemoteComputeTarget): Promise<SchemeCatalog> {
  if (executionTarget && !isTauriRuntime()) throw new Error("Remote compute requires Nimbus Studio desktop.");
  if (!isTauriRuntime()) return MOCK_SCHEME_CATALOG;
  const raw = await invoke<unknown>("scheme_catalog", {
    simulatorPath: binaryPath,
    executionTarget,
  });
  return validateSchemeCatalog(raw);
}

export async function probeRemoteCompute(executionTarget: RemoteComputeTarget): Promise<RemoteComputeProbe> {
  return invoke<RemoteComputeProbe>("probe_remote_compute", { executionTarget });
}

export interface SavedCm1SoundingProjection {
  schema: "nimbus.studio.saved-cm1-sounding-projection";
  schemaVersion: 1;
  canonicalPath: string;
  byteLength: number;
  levelCount: number;
}

/** Native-boundary save for a reviewed CM1 sounding projection. The Rust
 * command independently validates the complete Nimbus text contract before
 * atomically replacing the selected target. */
export async function saveCm1SoundingProjection(
  path: string,
  contents: string,
  sourcePath: string,
  sourceIsnd: number | null,
  reviewed: boolean,
): Promise<SavedCm1SoundingProjection> {
  if (!isTauriRuntime()) {
    throw new Error("reviewed sounding export requires Nimbus Studio desktop");
  }
  return invoke<SavedCm1SoundingProjection>("save_cm1_sounding_projection", {
    path,
    contents,
    sourcePath,
    sourceIsnd,
    reviewed,
  });
}

/** Pick an original CM1 input_sounding. No extension filter is intentional:
 * canonical CM1 files are commonly named `input_sounding` without one. */
export async function pickNativeCm1SoundingFile(): Promise<string | null> {
  if (!isTauriRuntime()) return null;
  const { open } = await import("@tauri-apps/plugin-dialog");
  const result = await open({
    multiple: false,
    directory: false,
    title: "Open native CM1 input_sounding",
  });
  return typeof result === "string" ? result : null;
}

/** Parse and bound-check a native CM1 input_sounding at the Rust boundary. */
export async function inspectNativeCm1Sounding(path: string): Promise<Cm1SoundingReport> {
  if (!isTauriRuntime()) {
    throw new Error("native CM1 sounding inspection requires Nimbus Studio desktop");
  }
  return invoke<Cm1SoundingReport>("inspect_cm1_sounding", { path });
}

export async function getHardwareInfo(): Promise<HardwareInfo> {
  if (!isTauriRuntime()) return mockHardware;
  return invoke<HardwareInfo>("hardware_info");
}

export async function getProcessMemoryInfo(): Promise<ProcessMemoryInfo | null> {
  if (!isTauriRuntime()) return null;
  return invoke<ProcessMemoryInfo>("process_memory_info");
}

export async function pickPngSavePath(defaultName: string): Promise<string | null> {
  if (!isTauriRuntime()) return null;
  const { save } = await import("@tauri-apps/plugin-dialog");
  return await save({ title: "Save current plot frame", defaultPath: defaultName, filters: [{ name: "PNG", extensions: ["png"] }] }) ?? null;
}

/** Native save dialog for a GIF. Returns the chosen path or null. */
export async function pickGifSavePath(
  defaultName: string,
): Promise<string | null> {
  if (!isTauriRuntime()) {
    console.info("[mock] pickGifSavePath called outside Tauri");
    return null;
  }
  const { save } = await import("@tauri-apps/plugin-dialog");
  const result = await save({
    title: "Save plot animation",
    defaultPath: defaultName,
    filters: [{ name: "GIF", extensions: ["gif"] }],
  });
  return result ?? null;
}

export async function pickViewpointsSavePath(defaultName: string): Promise<string | null> {
  if (!isTauriRuntime()) return null;
  const { save } = await import("@tauri-apps/plugin-dialog");
  return await save({ title: "Save portable viewpoints", defaultPath: defaultName,
    filters: [{ name: "Interactive HTML viewpoints", extensions: ["html"] }] }) ?? null;
}

/** Native save dialog for a generated MP4 movie. */
export async function pickMovieSavePath(
  defaultName: string,
): Promise<string | null> {
  if (!isTauriRuntime()) {
    console.info("[mock] pickMovieSavePath called outside Tauri");
    return null;
  }
  const { save } = await import("@tauri-apps/plugin-dialog");
  const result = await save({
    title: "Save plot movie",
    defaultPath: defaultName,
    filters: [{ name: "MP4", extensions: ["mp4"] }],
  });
  return result ?? null;
}

/** Local ffmpeg detection result. path == null means not found. */
export interface FfmpegInfo {
  path: string | null;
  version: string | null;
}

/** Probe for ffmpeg without encoding. Used by Settings to surface detection status. */
export async function ffmpegDetect(
  overridePath?: string,
): Promise<FfmpegInfo> {
  if (!isTauriRuntime()) {
    return { path: null, version: null };
  }
  return invoke<FfmpegInfo>("ffmpeg_detect", { overridePath });
}

/** Create a temp dir for one MP4 recording; returns the session id. */
export async function prepareEncodeSession(): Promise<string> {
  if (!isTauriRuntime()) {
    console.info("[mock] prepareEncodeSession");
    return "mock-session";
  }
  return invoke<string>("prepare_encode_session");
}

/** Append one base64 PNG frame to an encode session (one at a time to bound IPC payload). */
export async function appendEncodeFrame(
  sessionId: string,
  index: number,
  frameBase64: string,
): Promise<void> {
  if (!isTauriRuntime()) return;
  return invoke<void>("append_encode_frame", { sessionId, index, frameBase64 });
}

/** Mux frames into an MP4 at `outputPath`, then delete temp frames. Rejects with AppError on failure. */
export async function encodeMovieDir(
  sessionId: string,
  outputPath: string,
  fps: number,
  ffmpegPathOverride?: string,
): Promise<void> {
  if (!isTauriRuntime()) {
    console.info(`[mock] encodeMovieDir(${sessionId}, ${outputPath}, ${fps}fps)`);
    return;
  }
  return invoke<void>("encode_movie_dir", {
    sessionId,
    outputPath,
    fps,
    ffmpegPathOverride,
  });
}

/** Delete a session's temp frames without muxing (cancel/error path). Best-effort. */
export async function cleanupEncodeSession(sessionId: string): Promise<void> {
  if (!isTauriRuntime()) return;
  return invoke<void>("cleanup_encode_session", { sessionId });
}

/** Write bytes to disk at `path` via the Tauri fs plugin. */
export async function writeBinaryFile(
  path: string,
  bytes: Uint8Array,
): Promise<void> {
  if (!isTauriRuntime()) {
    console.info(`[mock] writeBinaryFile(${path}, ${bytes.byteLength} bytes)`);
    return;
  }
  const { writeFile } = await import("@tauri-apps/plugin-fs");
  await writeFile(path, bytes);
}

/** File extensions stageable into the Python interpreter (netCDF, HDF5, numpy). */
export const PY_DATA_FILE_EXTENSIONS = ["nc", "nc4", "cdf", "netcdf", "npy", "npz"] as const;

/** Native picker for data files to load inside the Python tab. Returns absolute
 *  paths, or an empty array on cancel. */
export async function pickDataFiles(): Promise<string[]> {
  if (!isTauriRuntime()) {
    console.info("[mock] pickDataFiles called outside Tauri");
    return [];
  }
  const { open } = await import("@tauri-apps/plugin-dialog");
  const result = await open({
    multiple: true,
    title: "Open data files in Python",
    filters: [{ name: "Data files", extensions: [...PY_DATA_FILE_EXTENSIONS] }],
  });
  if (result == null) return [];
  return Array.isArray(result) ? result.filter((entry) => typeof entry === "string") : [result];
}

/** Read a file as raw bytes. Requires `fs:allow-read-file` capability. */
export async function readBinaryFile(path: string): Promise<Uint8Array> {
  if (!isTauriRuntime()) {
    throw new Error("readBinaryFile requires the Tauri runtime");
  }
  const { readFile } = await import("@tauri-apps/plugin-fs");
  return readFile(path);
}

/** Byte length of a file without reading it (reject oversized files before the read). */
export async function fileByteLength(path: string): Promise<number> {
  if (!isTauriRuntime()) {
    throw new Error("fileByteLength requires the Tauri runtime");
  }
  const { stat } = await import("@tauri-apps/plugin-fs");
  const info = await stat(path);
  return info.size;
}

/** Native save dialog for Python-produced artefacts; filter derived from the artefact type. */
export async function pickExportSavePath(
  defaultName: string,
  filterLabel: string,
  extensions: readonly string[],
): Promise<string | null> {
  if (!isTauriRuntime()) {
    console.info(`[mock] pickExportSavePath(${defaultName})`);
    return null;
  }
  const { save } = await import("@tauri-apps/plugin-dialog");
  const result = await save({
    title: "Export from Python",
    defaultPath: defaultName,
    filters: extensions.length > 0
      ? [{ name: filterLabel, extensions: [...extensions] }]
      : undefined,
  });
  return result ?? null;
}

/** Native file picker for a plot spec. Returns the path or null on cancel. */
export async function pickSpecLoadPath(): Promise<string | null> {
  if (!isTauriRuntime()) {
    console.info("[mock] pickSpecLoadPath called outside Tauri");
    return null;
  }
  const { open } = await import("@tauri-apps/plugin-dialog");
  const result = await open({
    multiple: false,
    title: "Load plot spec",
    filters: [{ name: "Nimbus plot", extensions: ["json", "nimbus-plot.json"] }],
  });
  if (result == null) return null;
  return typeof result === "string" ? result : null;
}

/** Native save dialog for a plot spec. Default extension is .json. */
export async function pickSpecSavePath(defaultName: string): Promise<string | null> {
  if (!isTauriRuntime()) {
    console.info("[mock] pickSpecSavePath called outside Tauri");
    return null;
  }
  const { save } = await import("@tauri-apps/plugin-dialog");
  const result = await save({
    title: "Save plot spec",
    defaultPath: defaultName,
    filters: [{ name: "Nimbus plot", extensions: ["json"] }],
  });
  return result ?? null;
}

/** Read a plot-spec file as UTF-8 text. */
export async function readSpec(path: string): Promise<string> {
  if (!isTauriRuntime()) {
    return `{"spec_version":1,"name":"mock","source":{"kind":"field","field":"u"},"renderer":{"kind":"heatmap"}}`;
  }
  return invoke<string>("read_config", { path });
}

/** Persist a plot spec to disk (reuses write_config — same path+contents contract). */
export async function writeSpec(path: string, contents: string): Promise<void> {
  if (!isTauriRuntime()) {
    console.info(`[mock] writeSpec(${path}, ${contents.length} bytes)`);
    return;
  }
  return invoke<void>("write_config", { path, contents });
}

/** Native file picker for a .py snippet. Returns null on cancel. */
export async function pickPySnippetLoadPath(): Promise<string | null> {
  if (!isTauriRuntime()) {
    console.info("[mock] pickPySnippetLoadPath called outside Tauri");
    return null;
  }
  const { open } = await import("@tauri-apps/plugin-dialog");
  const result = await open({
    multiple: false,
    title: "Load Python snippet",
    filters: [{ name: "Python", extensions: ["py"] }],
  });
  if (result == null) return null;
  return typeof result === "string" ? result : null;
}

/** Native save dialog for a Python snippet. Default extension is .py. */
export async function pickPySnippetSavePath(
  defaultName: string,
): Promise<string | null> {
  if (!isTauriRuntime()) {
    console.info("[mock] pickPySnippetSavePath called outside Tauri");
    return null;
  }
  const { save } = await import("@tauri-apps/plugin-dialog");
  const result = await save({
    title: "Save Python snippet",
    defaultPath: defaultName,
    filters: [{ name: "Python", extensions: ["py"] }],
  });
  return result ?? null;
}

/** Read a .py snippet as UTF-8 text (reuses read_config). */
export async function readPySnippet(path: string): Promise<string> {
  if (!isTauriRuntime()) {
    return "# mock python\nresult = u[:, :, 0]\n";
  }
  return invoke<string>("read_config", { path });
}

/** Persist a .py snippet to disk (reuses write_config). */
export async function writePySnippet(
  path: string,
  contents: string,
): Promise<void> {
  if (!isTauriRuntime()) {
    console.info(`[mock] writePySnippet(${path}, ${contents.length} bytes)`);
    return;
  }
  return invoke<void>("write_config", { path, contents });
}

/* ---------- Plugin gallery ---------- */

/** Matches PluginEntry in src-tauri/src/commands/plugins.rs. */
export interface PluginEntry {
  name: string;
  path: string;
  sizeBytes: number;
  modifiedMs: number | null;
}

/** Resolve ~/.nimbus/plugins/, creating it on first call. */
export async function getPluginsDir(): Promise<string> {
  if (!isTauriRuntime()) {
    return "/__mock_plugins_dir__";
  }
  return invoke<string>("plugins_dir");
}

/** List .py files in the plugins dir, newest first. Empty list when dir is empty. */
export async function listPlugins(): Promise<PluginEntry[]> {
  if (!isTauriRuntime()) {
    return [];
  }
  return invoke<PluginEntry[]>("list_plugins");
}

/** Native picker for .yaml / .yml files. Returns the path or null. */
export async function pickYamlFile(): Promise<string | null> {
  if (!isTauriRuntime()) {
    console.info("[mock] pickYamlFile called outside Tauri");
    return null;
  }
  const { open } = await import("@tauri-apps/plugin-dialog");
  const result = await open({
    multiple: false,
    title: "Upload Nimbus config",
    filters: [{ name: "YAML", extensions: ["yaml", "yml"] }],
  });
  if (result == null) return null;
  return typeof result === "string" ? result : null;
}

/** Native save dialog for .yaml. Returns the chosen path or null. */
export async function pickYamlSavePath(
  defaultName: string,
): Promise<string | null> {
  if (!isTauriRuntime()) {
    console.info("[mock] pickYamlSavePath called outside Tauri");
    return null;
  }
  const { save } = await import("@tauri-apps/plugin-dialog");
  const result = await save({
    title: "Save Nimbus config",
    defaultPath: defaultName,
    filters: [{ name: "YAML", extensions: ["yaml", "yml"] }],
  });
  return result ?? null;
}

export interface CheckpointInspection {
  path: string;
  sidecarPath: string;
  schema: string;
  schemaVersion: number;
  checkpointBytes: number;
  checkpointFnv1a64: string;
  runId: string;
  parentRunId: string | null;
  parentCheckpoint: string | null;
  completedSteps: number;
  simulationTimeS: number;
  boundary: string;
  portableAcrossComputeMemoryTopology: boolean;
  integrityVerified: boolean;
  formatCompatible: boolean;
  scientificCompatibility: "verified_before_state_restore";
}

/** Select a checkpoint file. Path is not accepted until inspectCheckpoint verifies its sidecar. */
export async function pickCheckpointFile(): Promise<string | null> {
  if (!isTauriRuntime()) {
    console.info("[mock] pickCheckpointFile called outside Tauri");
    return null;
  }
  const { open } = await import("@tauri-apps/plugin-dialog");
  const result = await open({
    multiple: false,
    title: "Inspect Nimbus checkpoint",
    filters: [{ name: "Nimbus checkpoint", extensions: ["nimbus"] }],
  });
  return typeof result === "string" ? result : null;
}

/** Verify schema, step boundary, topology portability, byte count, and FNV-1a digest. */
export async function inspectCheckpoint(path: string, executionTarget?: RemoteComputeTarget): Promise<CheckpointInspection> {
  return invoke<CheckpointInspection>("inspect_checkpoint", { path, executionTarget });
}

/** Pick a save path for a sounding projection (write is done by saveCm1SoundingProjection). */
export async function pickSoundingSavePath(
  defaultName = "cm1-projected-sounding.txt",
): Promise<string | null> {
  if (!isTauriRuntime()) {
    console.info("[mock] pickSoundingSavePath called outside Tauri");
    return null;
  }
  const { save } = await import("@tauri-apps/plugin-dialog");
  return save({
    title: "Save reviewed Nimbus sounding",
    defaultPath: defaultName,
    filters: [{ name: "Nimbus sounding", extensions: ["txt", "sounding"] }],
  });
}

/* ---------- Event subscriptions ---------- */

export type Unsubscribe = () => void;

export async function onRunLine(
  cb: (event: LineEvent) => void,
): Promise<Unsubscribe> {
  if (!isTauriRuntime()) {
    return () => undefined;
  }
  const { listen } = await import("@tauri-apps/api/event");
  const handle = await listen<LineEvent>("run:line", (e) => cb(e.payload));
  return () => handle();
}

export async function onRunProgress(
  cb: (event: ProgressEvent) => void,
): Promise<Unsubscribe> {
  if (!isTauriRuntime()) return () => undefined;
  const { listen } = await import("@tauri-apps/api/event");
  const handle = await listen<ProgressEvent>("run:progress", (e) => cb(e.payload));
  return () => handle();
}

export async function onRunExit(
  cb: (event: ExitEvent) => void,
): Promise<Unsubscribe> {
  if (!isTauriRuntime()) return () => undefined;
  const { listen } = await import("@tauri-apps/api/event");
  const handle = await listen<ExitEvent>("run:exit", (e) => cb(e.payload));
  return () => handle();
}

export async function onRemoteReceiveChanged(cb: () => void): Promise<() => void> {
  if (!isTauriRuntime()) return () => {};
  const { listen } = await import("@tauri-apps/api/event");
  return listen("run:receive", cb);
}
