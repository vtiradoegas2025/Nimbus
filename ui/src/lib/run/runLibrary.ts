import type { QueuedRun, RunLibraryEntry } from "@/lib/core/types";

function normalizePath(path: string): string {
  return path.replace(/[\\/]+$/, "");
}

export function libraryEntryToRun(entry: RunLibraryEntry): QueuedRun {
  const terminalTime = entry.simulationTimeS ?? 0;
  const observedStatus = entry.provenance.runtimeHealth === "verified" ? entry.provenance.finalStatus : null;
  return {
    id: entry.id,
    caseName: entry.caseName,
    scheme: "klemp_wilhelmson_split_cartesian_cgrid",
    status: observedStatus === "completed" || observedStatus === "failed" || observedStatus === "aborted"
      ? observedStatus : entry.health === "ready"
        ? "completed"
        : entry.health === "partial"
          ? "aborted"
          : "failed",
    executionTarget: entry.executionTarget,
    remoteReceive: entry.remoteReceive,
    remoteTransferInProgress: entry.remoteReceive?.state === "receiving" && !entry.remoteReceive.retryAvailable,
    startedAtMs: entry.startedAtMs,
    endedAtMs: entry.modifiedAtMs,
    simTimeS: entry.simulationTimeS,
    step: entry.latestStep,
    velMax: null,
    durationS: terminalTime,
    outputDir: entry.outputDir,
    origin: "disk",
    libraryHealth: entry.health,
    libraryIssue: entry.issue,
    frameCount: entry.frameCount,
    fieldCount: entry.fieldCount,
    outputFormat: entry.outputFormat,
    modifiedAtMs: entry.modifiedAtMs,
    libraryScanLimited: entry.scanLimited,
    libraryScanLimit: entry.scanLimit,
    provenance: entry.provenance,
  };
}

/** Merge durable disk discovery with the current process registry. Output
 * directory is the identity join: a live row wins lifecycle fields while
 * retaining the disk qualification summary. */
export function mergeRunLibrary(
  liveRuns: QueuedRun[],
  libraryEntries: RunLibraryEntry[],
): QueuedRun[] {
  const diskByPath = new Map(
    libraryEntries.map((entry) => [normalizePath(entry.outputDir), entry] as const),
  );
  const merged: QueuedRun[] = liveRuns.map((run): QueuedRun => {
    const disk = run.outputDir ? diskByPath.get(normalizePath(run.outputDir)) : undefined;
    if (!disk) return { ...run, origin: "live" as const };
    diskByPath.delete(normalizePath(disk.outputDir));
    return {
      ...run,
      executionTarget: run.executionTarget === undefined ? disk.executionTarget : run.executionTarget,
      remoteReceive: run.remoteReceive ?? disk.remoteReceive,
      remoteTransferInProgress: run.remoteTransferInProgress
        || (run.remoteReceive ?? disk.remoteReceive)?.state === "receiving"
          && !(run.remoteReceive ?? disk.remoteReceive)?.retryAvailable,
      origin: "live" as const,
      libraryHealth: disk.health,
      libraryIssue: disk.issue,
      frameCount: disk.frameCount,
      fieldCount: disk.fieldCount,
      outputFormat: disk.outputFormat,
      modifiedAtMs: disk.modifiedAtMs,
      libraryScanLimited: disk.scanLimited,
      libraryScanLimit: disk.scanLimit,
      provenance: disk.provenance,
    };
  });
  merged.push(...[...diskByPath.values()].map(libraryEntryToRun));
  return merged.sort(
    (a, b) =>
      (b.startedAtMs ?? b.modifiedAtMs ?? 0) -
        (a.startedAtMs ?? a.modifiedAtMs ?? 0) || a.id.localeCompare(b.id),
  );
}

export function filterRunLibrary(runs: QueuedRun[], query: string): QueuedRun[] {
  const terms = query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
  if (terms.length === 0) return runs;
  return runs.filter((run) => {
    const searchable = [
      run.caseName,
      run.outputDir,
      run.scheme,
      run.status,
      run.libraryHealth,
      run.outputFormat,
      run.provenance?.health,
      run.provenance?.runtimeHealth,
      run.provenance?.requestedBackend,
      run.provenance?.requestedFallbackAllowed === null
        ? undefined
        : run.provenance?.requestedFallbackAllowed
          ? "fallback allowed"
          : "fallback disabled",
      run.provenance?.requestedOutputFormat,
      run.provenance?.requestedExecutionMode,
      run.provenance?.actualBackend,
      run.provenance?.finalStatus,
      run.provenance?.soundingCanonicalPath,
      run.provenance?.soundingRequestedPath,
      run.provenance?.soundingDigest,
      ...Object.values(run.provenance?.requestedSchemes ?? {}),
      ...Object.values(run.provenance?.actualSchemes ?? {}),
    ]
      .filter((value): value is string => Boolean(value))
      .join(" ")
      .toLocaleLowerCase();
    return terms.every((term) => searchable.includes(term));
  });
}
