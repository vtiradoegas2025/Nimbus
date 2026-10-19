/*
 * A Python run's output, held outside the component that started it.
 *
 * Studio routes with a switch (App.tsx), so leaving the Plots page unmounts the
 * panel while the worker keeps running. Writing through these functions rather
 * than setState is what lets a run's result land with nobody mounted.
 */

import type { RunResult } from "@/lib/platform/pyodideRuntime";
import type { StagedFile } from "@/lib/platform/pyodideFiles";
import type { SkippedField } from "@/lib/studio/pythonFieldRefs";
import type { StopStage } from "@/lib/studio/pythonStop";

/** Which frame a result was computed from, or null when nothing was open. */
export interface ResultFrame {
  stepIndex: number;
  simulationTimeS: number | null;
  publicationGeneration: string | null;
}

/** One cell's last run, kept per cell so running another does not discard it. */
export interface CellRun {
  readonly result: RunResult;
  /** The cell text this came from; positions shift, so the index alone cannot
   *  tell whether this is still that cell's output. */
  readonly code: string;
  /** Null when the cell ran with no dataset open, and so plots nothing from one. */
  readonly frame: ResultFrame | null;
}

/** The tail of a running cell's output, shown while it works. */
export interface LiveStream {
  stdout: string;
  stderr: string;
  /** Set once either stream has dropped its head to stay inside the cap. */
  truncated: boolean;
}

export interface PythonRunSession {
  readonly running: boolean;
  /** Co-located with `running` so both move atomically; split state caused inverted stop messages. */
  readonly stopStage: StopStage;
  /** Monotonically increasing count of finished runs. Tests wait for "n+1" rather than "not running". */
  readonly completedRuns: number;
  /** Dataset the stored results were computed against. Results are cleared on dataset change only, not on remount. */
  readonly datasetToken: string | null;
  /** Buffer that owns every retained/live output field below. Switching tabs
   *  must never reinterpret one script's cell indices as another script's. */
  readonly outputBufferId: string | null;
  readonly cellRuns: ReadonlyMap<number, CellRun>;
  readonly displayedCell: number | null;
  readonly lastRunCellIndices: readonly number[];
  readonly liveStream: LiveStream | null;
  /** Fields excluded by the auto-load budget. Reported above the NameError they will cause. */
  readonly skippedFields: readonly SkippedField[];
  readonly ioStatus: string | null;
  readonly loadErr: string | null;
  /** Set when the interpreter itself is gone; only a restart clears it. */
  readonly fatalError: string | null;
  /** Host-file metadata mirrors the pane-scoped interpreter filesystem and
   * survives route unmounts with it. */
  readonly stagedFiles: readonly StagedFile[];
}

const EMPTY: PythonRunSession = Object.freeze({
  running: false,
  stopStage: "idle" as StopStage,
  completedRuns: 0,
  datasetToken: null,
  outputBufferId: null,
  cellRuns: new Map<number, CellRun>(),
  displayedCell: null,
  lastRunCellIndices: [],
  liveStream: null,
  skippedFields: [],
  ioStatus: null,
  loadErr: null,
  fatalError: null,
  stagedFiles: [],
});

/** LRU capacity; eviction must never touch a session whose run is still in flight. */
const MAX_SESSIONS = 12;

const sessions = new Map<string, PythonRunSession>();
const listeners = new Map<string, Set<() => void>>();

/** Identity of one panel's run state: which project, which pane. */
export function sessionKey(projectId: string | null, paneId: string): string {
  return `${projectId ?? "none"}::${paneId}`;
}

/** Current session as a stable reference. useSyncExternalStore re-renders forever on new object identity. */
export function readSession(key: string): PythonRunSession {
  return sessions.get(key) ?? EMPTY;
}

export function patchSession(
  key: string,
  patch: Partial<PythonRunSession> | ((previous: PythonRunSession) => Partial<PythonRunSession>),
): void {
  const previous = readSession(key);
  const delta = typeof patch === "function" ? patch(previous) : patch;
  let changed = false;
  for (const field of Object.keys(delta) as Array<keyof PythonRunSession>) {
    if (!Object.is(previous[field], delta[field])) {
      changed = true;
      break;
    }
  }
  // A no-op patch must not notify. The run loop writes the same `running: true`
  // on every cell of a sequence, and waking every subscriber for a value that
  // did not move is a render per cell for nothing.
  if (!changed) return;

  sessions.set(key, { ...previous, ...delta });
  evictCold(key);
  for (const listener of listeners.get(key) ?? []) listener();
}

function evictCold(keep: string): void {
  if (sessions.size <= MAX_SESSIONS) return;
  for (const [key, session] of sessions) {
    if (sessions.size <= MAX_SESSIONS) return;
    // Insertion order is oldest-first, and a Map re-set does not reorder — so
    // this walks least-recently-created first, skipping anything still live.
    if (key === keep || session.running || listeners.get(key)?.size) continue;
    sessions.delete(key);
  }
}

export function subscribeSession(key: string, listener: () => void): () => void {
  let set = listeners.get(key);
  if (!set) {
    set = new Set();
    listeners.set(key, set);
  }
  set.add(listener);
  return () => {
    const current = listeners.get(key);
    if (!current) return;
    current.delete(listener);
    if (current.size === 0) listeners.delete(key);
  };
}

/** Drop everything. Exists for tests, which must not inherit each other's runs. */
export function resetSessions(): void {
  sessions.clear();
  listeners.clear();
}
