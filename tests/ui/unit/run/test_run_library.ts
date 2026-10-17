import { filterRunLibrary, libraryEntryToRun, mergeRunLibrary } from "../../../../ui/src/lib/run/runLibrary.ts";
import type { QueuedRun, RunLibraryEntry } from "../../../../ui/src/lib/core/types.ts";

let passed = 0;
function check(condition: unknown, message: string) {
  if (!condition) throw new Error(message);
  passed += 1;
}

const ready: RunLibraryEntry = {
  id: "disk-stable",
  caseName: "classic-supercell",
  outputDir: "/runs/1700000000000-classic-supercell/",
  health: "ready",
  issue: null,
  startedAtMs: 1700000000000,
  modifiedAtMs: 1700000005000,
  frameCount: 12,
  latestStep: 110,
  simulationTimeS: 900,
  fieldCount: 18,
  outputFormat: "netCDF",
  scanLimited: false,
  scanLimit: 512,
  provenance: {
    health: "verified",
    issue: null,
    runtimeHealth: "verified",
    runtimeIssue: null,
    runId: "run-provenance",
    createdUnixMs: 1700000000000,
    executablePath: "/opt/nimbus/bin/nimbus_sim",
    executableDigest: "fnv1a64:1111111111111111",
    configurationDigest: "fnv1a64:2222222222222222",
    catalogFingerprint: "fnv1a64:catalog",
    requestedBackend: "cpu",
    requestedFallbackAllowed: false,
    requestedOutputFormat: "netcdf",
    requestedExecutionMode: "qualified",
    requestedValidationMode: "error",
    requestedSchemes: { microphysics: "kessler" },
    soundingRequestedPath: "/inputs/wk.sounding",
    soundingCanonicalPath: "/verified/inputs/wk.sounding",
    soundingDigest: "fnv1a64:4444444444444444",
    soundingIssue: null,
    runtimeActualsAvailable: true,
    actualBackend: "cpu",
    fallbackActive: false,
    actualSchemes: { microphysics: "kessler" },
    finalStatus: "completed",
    exitCode: 0,
    processStartedUnixMs: 1700000000001,
    processEndedUnixMs: 1700000005000,
    completeFrameCount: 12,
    actualOutputFormat: "netCDF",
    verifiedCheckpointCount: 1,
    latestCheckpointPath: "/runs/checkpoints/checkpoint_step_000000110.nimbus",
  },
};

const converted = libraryEntryToRun(ready);
check(converted.id === "disk-stable", "disk identity is preserved");
check(converted.origin === "disk", "disk origin is explicit");
check(converted.status === "completed", "qualified disk run maps to completed");
check(converted.durationS === 900, "last simulated time gives a truthful terminal duration");
check(converted.frameCount === 12 && converted.fieldCount === 18, "summary counts survive mapping");
check(converted.provenance?.health === "verified", "validated provenance survives mapping");
check(converted.provenance?.runtimeHealth === "verified", "validated runtime linkage survives mapping");
check(converted.provenance?.actualBackend === "cpu", "observed runtime backend survives mapping");
check(converted.provenance?.requestedFallbackAllowed === false, "requested fallback policy survives mapping");
check(converted.provenance?.soundingCanonicalPath === "/verified/inputs/wk.sounding", "sounding identity survives mapping");
check(libraryEntryToRun({ ...ready, provenance: { ...ready.provenance, finalStatus: "aborted" } }).status === "aborted",
  "complete readable frames do not relabel an observed aborted computation as completed");
check(libraryEntryToRun({ ...ready, provenance: { ...ready.provenance, finalStatus: "failed" } }).status === "failed",
  "valid output availability remains separate from observed failed execution");

const remoteTarget = { id: "node", name: "Linux compute", host: "compute.example", binaryPath: "/opt/nimbus/bin/nimbus_sim", workDirectory: "/srv/nimbus" };
const receiveFailure = { state: "failed" as const, error: "Connection interrupted", retryAvailable: true, updatedUnixMs: 1700000006000 };
const pendingRemote = { ...ready, id: "remote-no-frames", frameCount: 0, fieldCount: 0, health: "invalid" as const,
  executionTarget: remoteTarget, remoteReceive: receiveFailure };
const recovery = libraryEntryToRun(pendingRemote);
check(recovery.remoteReceive === receiveFailure && recovery.executionTarget === remoteTarget,
  "zero-frame remote archives preserve their original compute target and retry receipt");
check(recovery.status === "completed" && !recovery.remoteTransferInProgress,
  "a failed transfer does not change a verified completed computation into an active job");
check(libraryEntryToRun({ ...pendingRemote, remoteReceive: { ...receiveFailure, state: "receiving", retryAvailable: false } }).remoteTransferInProgress,
  "active receiving is visible independently of terminal computation status");
check(!libraryEntryToRun({ ...pendingRemote, remoteReceive: { ...receiveFailure, state: "receiving", retryAvailable: true } }).remoteTransferInProgress,
  "an interrupted restart receipt does not advertise an active transfer");

const live: QueuedRun = {
  ...converted,
  id: "run-live",
  outputDir: "/runs/1700000000000-classic-supercell",
  status: "running",
  origin: "live",
};
const currentReceipt = { ...receiveFailure, state: "receiving" as const, retryAvailable: false, updatedUnixMs: 1700000007000 };
const mergedReceive = mergeRunLibrary([{ ...live, remoteReceive: currentReceipt }], [{ ...ready, executionTarget: remoteTarget, remoteReceive: receiveFailure }])[0];
check(mergedReceive.remoteReceive === currentReceipt && mergedReceive.remoteTransferInProgress,
  "a stale library scan cannot replace the current live receiving state");
const archivedReceive = mergeRunLibrary([], [pendingRemote]);
check(archivedReceive.length === 1 && archivedReceive[0].remoteReceive?.retryAvailable,
  "a remote run stays discoverable and retryable before any step has been downloaded");
const invalid: RunLibraryEntry = {
  ...ready,
  id: "disk-invalid",
  caseName: "damaged-case",
  outputDir: "/runs/damaged-case",
  health: "invalid",
  issue: "No complete publications were found.",
  frameCount: 0,
  fieldCount: 0,
  outputFormat: "Unknown",
  provenance: {
    health: "missing",
    issue: "This run predates Studio provenance or was not launched by Studio.",
    runtimeHealth: "missing",
    runtimeIssue: "No Studio runtime completion record was found.",
    runId: null,
    createdUnixMs: null,
    executablePath: null,
    executableDigest: null,
    configurationDigest: null,
    catalogFingerprint: null,
    requestedBackend: null,
    requestedFallbackAllowed: null,
    requestedOutputFormat: null,
    requestedExecutionMode: null,
    requestedValidationMode: null,
    requestedSchemes: {},
    soundingRequestedPath: null,
    soundingCanonicalPath: null,
    soundingDigest: null,
    soundingIssue: "No sounding artifact was recorded.",
    runtimeActualsAvailable: false,
    actualBackend: null,
    fallbackActive: null,
    actualSchemes: {},
    finalStatus: null,
    exitCode: null,
    processStartedUnixMs: null,
    processEndedUnixMs: null,
    completeFrameCount: null,
    actualOutputFormat: null,
    verifiedCheckpointCount: null,
    latestCheckpointPath: null,
  },
};
const merged = mergeRunLibrary([live], [ready, invalid]);
check(merged.length === 2, "live and disk rows deduplicate by normalized output path");
check(merged.find((run) => run.id === "run-live")?.status === "running", "live lifecycle wins merge");
check(merged.find((run) => run.id === "run-live")?.libraryHealth === "ready", "disk health enriches live row");
check(merged.some((run) => run.id === "disk-invalid"), "invalid run stays discoverable");

check(filterRunLibrary(merged, "classic netcdf").length === 1, "multi-term search matches case and format");
check(filterRunLibrary(merged, "damaged invalid").length === 1, "health is searchable");
check(filterRunLibrary(merged, "qualified kessler").length === 1, "provenance facts are searchable");
check(filterRunLibrary(merged, "completed cpu").length === 1, "runtime evidence is searchable");
check(filterRunLibrary(merged, "verified wk.sounding").length === 1, "sounding provenance is searchable");
check(filterRunLibrary(merged, "  ").length === 2, "blank search preserves the library");
check(filterRunLibrary(merged, "unfindable").length === 0, "unmatched search returns empty");

console.log(`run library: ${passed} assertions passed`);
