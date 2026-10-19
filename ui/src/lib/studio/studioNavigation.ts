import type { PageId } from "@/lib/core/types";

export const STUDIO_NAVIGATION_STORAGE_KEY = "nimbus-studio:navigation:v1";

export interface StudioNavigationState {
  page: PageId;
  selectedRunId: string | null;
}

interface ReadStorage {
  getItem(key: string): string | null;
}

interface WriteStorage {
  setItem(key: string, value: string): void;
}

const PAGE_IDS = new Set<PageId>(["projects", "config", "runs", "plots", "settings"]);
const DEFAULT_STATE: StudioNavigationState = { page: "config", selectedRunId: null };

function isPageId(value: unknown): value is PageId {
  return typeof value === "string" && PAGE_IDS.has(value as PageId);
}

function boundedRunId(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 && value.length <= 256 ? value : null;
}

/** Restore only a known destination and a bounded opaque run identity. A
 * malformed or future record cannot strand Studio outside a reachable page. */
export function loadStudioNavigation(storage?: ReadStorage): StudioNavigationState {
  if (!storage) return { ...DEFAULT_STATE };
  try {
    const serialized = storage.getItem(STUDIO_NAVIGATION_STORAGE_KEY);
    if (!serialized) return { ...DEFAULT_STATE };
    const value: unknown = JSON.parse(serialized);
    if (!value || typeof value !== "object") return { ...DEFAULT_STATE };
    const record = value as Record<string, unknown>;
    if (record.schema !== "nimbus.studio.navigation" || record.schemaVersion !== 1 || !isPageId(record.page)) {
      return { ...DEFAULT_STATE };
    }
    return { page: record.page, selectedRunId: boundedRunId(record.selectedRunId) };
  } catch {
    return { ...DEFAULT_STATE };
  }
}

export function saveStudioNavigation(
  storage: WriteStorage | undefined,
  state: StudioNavigationState,
): void {
  if (!storage) return;
  try {
    storage.setItem(STUDIO_NAVIGATION_STORAGE_KEY, JSON.stringify({
      schema: "nimbus.studio.navigation",
      schemaVersion: 1,
      page: isPageId(state.page) ? state.page : DEFAULT_STATE.page,
      selectedRunId: boundedRunId(state.selectedRunId),
    }));
  } catch {
    // Navigation remains usable in memory when persistence is unavailable.
  }
}

/** A removed or relocated run must not leave Run Archive rendering stale
 * identity. The page remains selected so the empty archive teaches recovery. */
export function reconcileStudioNavigationRuns(
  state: StudioNavigationState,
  availableRunIds: readonly string[],
): StudioNavigationState {
  if (!state.selectedRunId || availableRunIds.includes(state.selectedRunId)) return state;
  return { ...state, selectedRunId: null };
}

/** Default run selection: active runs first, then newest. Returns null for an empty list. */
export function defaultArchiveRunId(
  runs: readonly { id: string; status: string; startedAtMs: number | null }[],
): string | null {
  if (runs.length === 0) return null;
  const newestFirst = (
    a: { startedAtMs: number | null },
    b: { startedAtMs: number | null },
  ) => (b.startedAtMs ?? -Infinity) - (a.startedAtMs ?? -Infinity);
  // Order of interest, not order of arrival: an active run outranks a queued
  // one, which outranks anything already terminal.
  for (const status of ["running", "queued"]) {
    const match = runs.filter((run) => run.status === status).sort(newestFirst)[0];
    if (match) return match.id;
  }
  return [...runs].sort(newestFirst)[0]?.id ?? null;
}

export function preferredCaseRunId(
  runs: readonly {
    id: string;
    caseName: string;
    outputDir?: string | null;
    libraryHealth?: "ready" | "partial" | "invalid";
    startedAtMs: number | null;
    modifiedAtMs?: number | null;
  }[],
  caseName: string,
): string | null {
  const expected = caseName.trim().toLocaleLowerCase();
  if (!expected) return null;
  return runs
    .filter((run) => run.caseName.toLocaleLowerCase() === expected
      && Boolean(run.outputDir)
      && run.libraryHealth !== "invalid")
    .sort((left, right) =>
      (right.startedAtMs ?? right.modifiedAtMs ?? -Infinity)
      - (left.startedAtMs ?? left.modifiedAtMs ?? -Infinity)
      || left.id.localeCompare(right.id))[0]?.id ?? null;
}
