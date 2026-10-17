import { useCallback, useEffect, useRef, useState } from "react";
import type {
  ExitEvent,
  HardwareInfo,
  ProgressEvent,
  QueuedRun,
  RunLibraryEntry,
  RunStatus,
  SimulatorStatus,
} from "./types";
import {
  deleteRun,
  discoverRunLibrary,
  getHardwareInfo,
  getSimulatorStatus,
  listRuns,
  onRunExit,
  onRunProgress,
  onRemoteReceiveChanged,
} from "@/lib/platform/tauri";
import { mergeRunLibrary } from "@/lib/run/runLibrary";
import { REMOTE_RECEIVE_CHANGE_EVENT } from "@/lib/platform/remoteCompute";
import {
  effectiveRunsDirectory,
  loadStudioSettings,
  STUDIO_SETTINGS_CHANGE_EVENT,
} from "@/lib/studio/studioSettings";

/* useSimulator polls Tauri once on mount. useRuns seeds from list_runs then
 * merges live progress/exit events with a 3s re-sync poll in the background. */

export interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

const RESYNC_INTERVAL_MS = 3000;
const LIBRARY_RESYNC_INTERVAL_MS = 30_000;

export function useSimulator(): AsyncState<SimulatorStatus> {
  const [state, setState] = useState<AsyncState<SimulatorStatus>>({
    data: null,
    loading: true,
    error: null,
  });

  useEffect(() => {
    let cancelled = false;
    let requestGeneration = 0;

    const refresh = async () => {
      const generation = ++requestGeneration;
      const { binaryPathOverride } = loadStudioSettings();
      try {
        const data = await getSimulatorStatus(binaryPathOverride || undefined);
        if (!cancelled && generation === requestGeneration) {
          setState({ data, loading: false, error: null });
        }
      } catch (e) {
        if (!cancelled && generation === requestGeneration) {
          setState({
            data: null,
            loading: false,
            error: e instanceof Error ? e.message : String(e),
          });
        }
      }
    };

    void refresh();
    const onSettingsChange = () => void refresh();
    window.addEventListener(STUDIO_SETTINGS_CHANGE_EVENT, onSettingsChange);
    return () => {
      cancelled = true;
      window.removeEventListener(STUDIO_SETTINGS_CHANGE_EVENT, onSettingsChange);
    };
  }, []);

  return state;
}

export function useHardware(): AsyncState<HardwareInfo> {
  const [state, setState] = useState<AsyncState<HardwareInfo>>({
    data: null,
    loading: true,
    error: null,
  });

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await getHardwareInfo();
        if (!cancelled) setState({ data, loading: false, error: null });
      } catch (e) {
        if (!cancelled)
          setState({
            data: null,
            loading: false,
            error: e instanceof Error ? e.message : String(e),
          });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return state;
}

/** Persisted drag-adjustable width. Pointer motion stays in memory; durable writes are coalesced. */
export function useResizableWidth(
  storageKey: string,
  initial: number,
  min: number,
  max: number,
): readonly [number, (deltaPx: number) => void] {
  const [width, setWidth] = useState<number>(() => {
    if (typeof window === "undefined") return initial;
    try {
      const raw = window.localStorage.getItem(storageKey);
      const n = raw == null ? NaN : Number.parseFloat(raw);
      if (Number.isFinite(n)) return Math.min(max, Math.max(min, n));
    } catch {
      // localStorage may be blocked by quota or sandbox; fall through.
    }
    return initial;
  });

  const widthRef = useRef(width);
  widthRef.current = width;
  const persistencePendingRef = useRef(false);

  useEffect(() => {
    if (!persistencePendingRef.current) return;
    const id = window.setTimeout(() => {
      try {
        window.localStorage.setItem(storageKey, String(widthRef.current));
        persistencePendingRef.current = false;
      } catch {
        // Persistence is best-effort. Width remains in-memory either way.
      }
    }, 150);
    return () => window.clearTimeout(id);
  }, [storageKey, width]);

  useEffect(() => {
    const flushPendingWidth = () => {
      if (!persistencePendingRef.current) return;
      try {
        window.localStorage.setItem(storageKey, String(widthRef.current));
        persistencePendingRef.current = false;
      } catch {
        // Persistence is best-effort. Width remains in-memory either way.
      }
    };
    window.addEventListener("pagehide", flushPendingWidth);
    return () => {
      window.removeEventListener("pagehide", flushPendingWidth);
      flushPendingWidth();
    };
  }, [storageKey]);

  const adjust = useCallback(
    (deltaPx: number) => {
      const next = Math.min(max, Math.max(min, widthRef.current + deltaPx));
      if (next === widthRef.current) return;
      widthRef.current = next;
      persistencePendingRef.current = true;
      setWidth(next);
    },
    [min, max],
  );

  return [width, adjust] as const;
}

/** Which subsystem failed: library = archive scan; live = list_runs; action = user op refused. */
export type RunsErrorScope = "library" | "live" | "action";

export interface RunsError {
  scope: RunsErrorScope;
  /** The underlying message. Always surfaced — a fixed banner that discards it
   *  leaves the user with no way to tell which failure they are looking at. */
  message: string;
}

export interface UseRunsResult extends Omit<AsyncState<QueuedRun[]>, "error"> {
  error: RunsError | null;
  /** Remove a run from the registry. Optimistically drops it from local
   *  state, then calls the Tauri delete_run command. On failure the
   *  next resync (3s) will re-add it. Refused server-side for running
   *  runs — the UI should hide its delete button in that case. */
  removeRun: (id: string) => Promise<void>;
}

type RunsState = Omit<AsyncState<QueuedRun[]>, "error"> & { error: RunsError | null };

const runsError = (scope: RunsErrorScope, cause: unknown): RunsError => ({
  scope,
  message: cause instanceof Error ? cause.message : String(cause),
});

function errorsEqual(a: RunsError | null, b: RunsError | null): boolean {
  return a === b || (a?.scope === b?.scope && a?.message === b?.message);
}

/** Preserve the previous array when a poll returns the same run snapshot. */
export function runSnapshotsEqual(
  previous: readonly QueuedRun[] | null,
  next: readonly QueuedRun[] | null,
): boolean {
  if (previous === next) return true;
  if (previous == null || next == null || previous.length !== next.length) return false;
  for (let index = 0; index < previous.length; index++) {
    const a = previous[index];
    const b = next[index];
    if (a === b) continue;
    // Run records are JSON-shaped IPC data. Stringification also covers the
    // optional provenance object without letting a newly allocated but
    // identical poll result fan out through every queue/detail consumer.
    if (JSON.stringify(a) !== JSON.stringify(b)) return false;
  }
  return true;
}

/** Apply one progress event, retaining the array when it carries no new data. */
export function applyRunProgressSnapshot(
  runs: QueuedRun[],
  evt: ProgressEvent,
): QueuedRun[] {
  const index = runs.findIndex((run) => run.id === evt.runId);
  if (index < 0) return runs;
  const current = runs[index];
  const step = evt.progress.step ?? current.step;
  const velMax = evt.progress.velMax ?? current.velMax;
  if (
    current.simTimeS === evt.progress.simTimeS &&
    current.step === step &&
    current.velMax === velMax
  ) return runs;
  const next = runs.slice();
  next[index] = {
    ...current,
    simTimeS: evt.progress.simTimeS,
    step,
    velMax,
  };
  return next;
}

export function useRuns(archiveVisible = true): UseRunsResult {
  const [state, setState] = useState<RunsState>({
    data: null,
    loading: true,
    error: null,
  });

  const archiveVisibleRef = useRef(archiveVisible);
  archiveVisibleRef.current = archiveVisible;
  const requestResyncRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    let cancelled = false;
    let progressUnsubscribe: (() => void) | undefined;
    let exitUnsubscribe: (() => void) | undefined;
    let receiveUnsubscribe: (() => void) | undefined;
    let libraryCache: RunLibraryEntry[] = [];
    let libraryRoot: string | null = null;
    let libraryError: RunsError | null = null;
    let lastLibraryScanMs = 0;

    // Guard against overlapping scans: later scan wins, which may be a stale snapshot.
    let resyncInFlight = false;

    const resync = async () => {
      if (resyncInFlight) return;
      resyncInFlight = true;
      try {
        const live = await listRuns();
        const runsDirectory = effectiveRunsDirectory(loadStudioSettings().runsDirectory);
        const now = Date.now();
        if (archiveVisibleRef.current && (
          libraryRoot !== runsDirectory ||
          now - lastLibraryScanMs >= LIBRARY_RESYNC_INTERVAL_MS
        )) {
          libraryRoot = runsDirectory;
          lastLibraryScanMs = now;
          try {
            libraryCache = await discoverRunLibrary(runsDirectory);
            libraryError = null;
          } catch (e) {
            // A mistyped or unreadable archive root must not hide live runs.
            libraryCache = [];
            libraryError = runsError("library", e);
          }
        }
        const data = mergeRunLibrary(live, libraryCache);
        if (!cancelled) {
          setState((prev) =>
            !prev.loading &&
            runSnapshotsEqual(prev.data, data) &&
            errorsEqual(prev.error, libraryError)
              ? prev
              : { data, loading: false, error: libraryError },
          );
        }
      } catch (e) {
        // listRuns() failed: the LIVE registry is what is unavailable here, so
        // this must not be reported as an archive-scan problem.
        if (!cancelled)
          setState((prev) => {
            const error = runsError("live", e);
            return !prev.loading && errorsEqual(prev.error, error)
              ? prev
              : { data: prev.data, loading: false, error };
          });
      } finally {
        resyncInFlight = false;
      }
    };
    requestResyncRef.current = () => void resync();

    const applyProgress = (evt: ProgressEvent) => {
      if (cancelled) return;
      setState((prev) => {
        if (!prev.data) return prev;
        const next = applyRunProgressSnapshot(prev.data, evt);
        if (next === prev.data) return prev;
        return { ...prev, data: next };
      });
    };

    const applyExit = (evt: ExitEvent) => {
      if (cancelled) return;
      setState((prev) => {
        if (!prev.data) return prev;
        const index = prev.data.findIndex((run) => run.id === evt.runId);
        if (index < 0) return prev;
        const current = prev.data[index];
        if (current.status === evt.status && current.endedAtMs != null) return prev;
        const next = prev.data.slice();
        next[index] = {
          ...current,
          status: evt.status as RunStatus,
          // Freeze the wall clock immediately on exit; the next resync
          // will overwrite with the authoritative Rust-side timestamp,
          // but this avoids one extra second of ticking in the UI.
          endedAtMs: current.endedAtMs ?? Date.now(),
        };
        return { ...prev, data: next };
      });
    };

    (async () => {
      await resync();
      if (cancelled) return;
      // Unsubscribe immediately if we unmounted while the listen() promise was
      // resolving — otherwise the cleanup below ran with the var still
      // undefined and the listener leaks for the session.
      const up = await onRunProgress(applyProgress);
      if (cancelled) {
        up();
        return;
      }
      progressUnsubscribe = up;
      const ue = await onRunExit(applyExit);
      if (cancelled) {
        ue();
        return;
      }
      exitUnsubscribe = ue;
      const ur = await onRemoteReceiveChanged(() => { lastLibraryScanMs = 0; void resync(); });
      if (cancelled) ur(); else receiveUnsubscribe = ur;
    })();

    const pollVisibleDocument = () => {
      if (document.visibilityState !== "hidden") void resync();
    };
    const receiveChanged = () => { lastLibraryScanMs = 0; void resync(); };
    window.addEventListener(REMOTE_RECEIVE_CHANGE_EVENT, receiveChanged);
    const id = window.setInterval(pollVisibleDocument, RESYNC_INTERVAL_MS);
    document.addEventListener("visibilitychange", pollVisibleDocument);

    return () => {
      cancelled = true;
      window.clearInterval(id);
      document.removeEventListener("visibilitychange", pollVisibleDocument);
      window.removeEventListener(REMOTE_RECEIVE_CHANGE_EVENT, receiveChanged);
      requestResyncRef.current = null;
      progressUnsubscribe?.();
      exitUnsubscribe?.();
      receiveUnsubscribe?.();
    };
  }, []);

  useEffect(() => {
    if (archiveVisible) requestResyncRef.current?.();
  }, [archiveVisible]);

  const removeRun = useCallback(async (id: string) => {
    // Snapshot for rollback if the backend refuses the delete.
    let prevRecord: QueuedRun | undefined;
    setState((prev) => {
      if (!prev.data) return prev;
      prevRecord = prev.data.find((r) => r.id === id);
      return { ...prev, data: prev.data.filter((r) => r.id !== id) };
    });
    try {
      await deleteRun(id);
    } catch (e) {
      // Restore the optimistic removal so the user sees the record again
      // and can react to the error message.
      setState((prev) => {
        if (!prev.data || !prevRecord) return prev;
        if (prev.data.some((r) => r.id === id)) return prev;
        return {
          ...prev,
          data: [...prev.data, prevRecord].sort(
            (a, b) => (b.startedAtMs ?? 0) - (a.startedAtMs ?? 0),
          ),
          // The backend's refusal reason (e.g. "cannot delete a running run")
          // is the only useful text here; it must reach the user verbatim.
          error: runsError("action", e),
        };
      });
      throw e;
    }
  }, []);

  return { ...state, removeRun };
}

export interface VelMaxSample {
  /** Wall-clock unix ms when the sample was recorded. */
  wallMs: number;
  /** Simulated time at the sample. */
  simTimeS: number;
  /** [VEL-MAX] value. */
  velMax: number;
}

/** In-memory velMax ring-buffer (256 samples) for the currently-viewed run. */
export function useVelMaxHistory(runId: string | null): VelMaxSample[] {
  const [samples, setSamples] = useState<VelMaxSample[]>([]);

  useEffect(() => {
    if (!runId) {
      setSamples([]);
      return;
    }
    setSamples([]);
    let cancelled = false;
    let unsubscribe: (() => void) | undefined;

    const MAX_SAMPLES = 256;

    (async () => {
      const unsub = await onRunProgress((evt) => {
        if (cancelled) return;
        if (evt.runId !== runId) return;
        if (evt.progress.velMax == null) return;
        const sample: VelMaxSample = {
          wallMs: Date.now(),
          simTimeS: evt.progress.simTimeS,
          velMax: evt.progress.velMax,
        };
        setSamples((prev) => {
          const next = prev.concat(sample);
          if (next.length > MAX_SAMPLES) return next.slice(next.length - MAX_SAMPLES);
          return next;
        });
      });
      // Unmounted while listen() was resolving -> unsubscribe now, else the
      // cleanup already ran with `unsubscribe` undefined and this leaks.
      if (cancelled) unsub();
      else unsubscribe = unsub;
    })();

    return () => {
      cancelled = true;
      unsubscribe?.();
    };
  }, [runId]);

  return samples;
}

/** Wall-clock ticker at 1 Hz. Only ticks when active. */
export function useTicker1Hz(active: boolean): number {
  const [now, setNow] = useState<number>(() => Date.now());
  useEffect(() => {
    if (!active) {
      // Stamp a fresh "now" once on deactivation so any consumer that
      // froze at the last tick gets a final accurate value.
      setNow(Date.now());
      return;
    }
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [active]);
  return now;
}
