import {
  DATASET_SOURCE_SCHEMA,
  DATASET_SOURCE_VERSION,
  datasetSourceIdentity,
  datasetSourcePaths,
  type LocalDatasetSource,
} from "./datasetSources";

export const RECENT_DATASETS_SCHEMA = "nimbus.studio.recent-datasets" as const;
export const RECENT_DATASETS_VERSION = 1 as const;
export const RECENT_DATASETS_STORAGE_KEY = "nimbus:studio:recent-datasets:v1" as const;
export const RECENT_DATASETS_LIMIT = 12 as const;

const MAX_PATH_LENGTH = 4_096;
const MAX_LABEL_LENGTH = 512;

export type RecentDatasetSelection =
  | { kind: "nimbus_folder"; paths: readonly [string] }
  | { kind: "cm1_files"; paths: readonly string[] }
  | { kind: "cm1_case_folder"; paths: readonly [string] };

export interface RecentDatasetEntry {
  datasetIdentity: string;
  source: LocalDatasetSource;
  selection: RecentDatasetSelection;
  lastOpenedAt: string;
  pinned: boolean;
}

export interface RecentDatasetStore {
  schema: typeof RECENT_DATASETS_SCHEMA;
  schemaVersion: typeof RECENT_DATASETS_VERSION;
  savedAt: string;
  entries: readonly RecentDatasetEntry[];
}

type ReadStorage = Pick<Storage, "getItem">;
type WriteStorage = Pick<Storage, "setItem">;
type RemoveStorage = Pick<Storage, "removeItem">;

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isSafeText(value: unknown, maximumLength: number): value is string {
  return typeof value === "string"
    && value.length > 0
    && value.length <= maximumLength
    && value.trim() === value
    && !value.includes("\0");
}

function isContractTimestamp(value: unknown): value is string {
  if (typeof value !== "string") return false;
  const time = Date.parse(value);
  return Number.isFinite(time) && new Date(time).toISOString() === value;
}

function sameStrings(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

/** Parse and sanitize a source reference. Optional runtime inspection reports
 * are deliberately dropped: the recent list is a locator, not a data cache. */
function parseSource(value: unknown): LocalDatasetSource | null {
  if (!isRecord(value) || !isRecord(value.locator)) return null;
  const locator = value.locator;
  if (!isSafeText(value.displayName, MAX_LABEL_LENGTH)
    || !isSafeText(locator.path, MAX_PATH_LENGTH)
    || value.schema !== DATASET_SOURCE_SCHEMA
    || value.schemaVersion !== DATASET_SOURCE_VERSION
    || value.readOnly !== true) return null;

  let paths: string[] | undefined;
  if (locator.kind === "local_paths") {
    if (!Array.isArray(locator.paths)
      || locator.paths.length < 2
      || locator.paths.length > 1_000
      || !locator.paths.every((path) => isSafeText(path, MAX_PATH_LENGTH))) return null;
    paths = [...locator.paths];
    if (new Set(paths).size !== paths.length || paths[0] !== locator.path) return null;
  } else if (locator.kind !== "local_path" || locator.paths != null) {
    return null;
  }

  const validKind = value.kind === "nimbus_run"
    ? value.sourceModel === "nimbus"
      && value.nativeFormat === "nimbus_history"
      && locator.kind === "local_path"
    : value.kind === "cm1_netcdf"
      && value.sourceModel === "cm1"
      && value.nativeFormat === "netcdf";
  if (!validKind) return null;

  return {
    schema: DATASET_SOURCE_SCHEMA,
    schemaVersion: DATASET_SOURCE_VERSION,
    kind: value.kind as LocalDatasetSource["kind"],
    sourceModel: value.sourceModel as LocalDatasetSource["sourceModel"],
    nativeFormat: value.nativeFormat as LocalDatasetSource["nativeFormat"],
    locator: paths
      ? { kind: "local_paths", path: locator.path, paths }
      : { kind: "local_path", path: locator.path },
    displayName: value.displayName,
    readOnly: true,
  };
}

function parseSelection(value: unknown, source: LocalDatasetSource): RecentDatasetSelection | null {
  if (!isRecord(value) || !Array.isArray(value.paths)) return null;
  const paths = value.paths;
  if (paths.length === 0
    || paths.length > 1_000
    || !paths.every((path) => isSafeText(path, MAX_PATH_LENGTH))
    || new Set(paths).size !== paths.length) return null;

  if (value.kind === "nimbus_folder") {
    return source.kind === "nimbus_run"
      && paths.length === 1
      && paths[0] === source.locator.path
      ? { kind: "nimbus_folder", paths: [paths[0]] }
      : null;
  }
  if (value.kind === "cm1_files") {
    return source.kind === "cm1_netcdf" && sameStrings(paths, datasetSourcePaths(source))
      ? { kind: "cm1_files", paths: [...paths] }
      : null;
  }
  if (value.kind === "cm1_case_folder") {
    return source.kind === "cm1_netcdf" && paths.length === 1
      ? { kind: "cm1_case_folder", paths: [paths[0]] }
      : null;
  }
  return null;
}

function parseEntry(value: unknown): RecentDatasetEntry | null {
  if (!isRecord(value)) return null;
  const source = parseSource(value.source);
  if (!source) return null;
  const selection = parseSelection(value.selection, source);
  if (!selection
    || typeof value.pinned !== "boolean"
    || !isContractTimestamp(value.lastOpenedAt)
    || typeof value.datasetIdentity !== "string"
    || value.datasetIdentity !== datasetSourceIdentity(source)) return null;
  return {
    datasetIdentity: value.datasetIdentity,
    source,
    selection,
    lastOpenedAt: value.lastOpenedAt,
    pinned: value.pinned,
  };
}

function sanitizedSource(source: LocalDatasetSource): LocalDatasetSource {
  const parsed = parseSource(source);
  if (!parsed) throw new Error("Cannot remember an invalid dataset source.");
  return parsed;
}

function selectionForSource(source: LocalDatasetSource): RecentDatasetSelection {
  if (source.kind === "nimbus_run") return { kind: "nimbus_folder", paths: [source.locator.path] };
  const caseDirectory = source.caseInventory?.directory;
  if (caseDirectory && isSafeText(caseDirectory, MAX_PATH_LENGTH)) {
    return { kind: "cm1_case_folder", paths: [caseDirectory] };
  }
  return { kind: "cm1_files", paths: [...datasetSourcePaths(source)] };
}

function newestFirst(left: RecentDatasetEntry, right: RecentDatasetEntry): number {
  if (left.pinned !== right.pinned) return left.pinned ? -1 : 1;
  return Date.parse(right.lastOpenedAt) - Date.parse(left.lastOpenedAt)
    || left.datasetIdentity.localeCompare(right.datasetIdentity);
}

function bounded(entries: readonly RecentDatasetEntry[], limit: number = RECENT_DATASETS_LIMIT): RecentDatasetEntry[] {
  const safeLimit = Number.isInteger(limit) && limit > 0
    ? Math.min(limit, RECENT_DATASETS_LIMIT)
    : RECENT_DATASETS_LIMIT;
  return [...entries].sort(newestFirst).slice(0, safeLimit);
}

export function createRecentDatasetEntry(
  source: LocalDatasetSource,
  options: { openedAt?: string; pinned?: boolean } = {},
): RecentDatasetEntry {
  const safeSource = sanitizedSource(source);
  const lastOpenedAt = options.openedAt ?? new Date().toISOString();
  if (!isContractTimestamp(lastOpenedAt)) throw new Error("Recent dataset timestamps must be canonical ISO-8601 values.");
  return {
    datasetIdentity: datasetSourceIdentity(safeSource),
    source: safeSource,
    selection: selectionForSource(source),
    lastOpenedAt,
    pinned: options.pinned ?? false,
  };
}

/** Add or touch one stable identity. An existing pin survives normal reopen. */
export function rememberRecentDataset(
  entries: readonly RecentDatasetEntry[],
  source: LocalDatasetSource,
  options: { openedAt?: string; pinned?: boolean; limit?: number } = {},
): RecentDatasetEntry[] {
  const candidate = createRecentDatasetEntry(source, options);
  const existing = entries.find((entry) => entry.datasetIdentity === candidate.datasetIdentity);
  const updated = existing && options.pinned == null
    ? { ...candidate, pinned: existing.pinned }
    : candidate;
  return bounded([
    updated,
    ...entries.filter((entry) => entry.datasetIdentity !== updated.datasetIdentity),
  ], options.limit);
}

export function setRecentDatasetPinned(
  entries: readonly RecentDatasetEntry[],
  datasetIdentity: string,
  pinned: boolean,
): RecentDatasetEntry[] {
  return bounded(entries.map((entry) => entry.datasetIdentity === datasetIdentity
    ? { ...entry, pinned }
    : entry));
}

export function forgetRecentDataset(
  entries: readonly RecentDatasetEntry[],
  datasetIdentity: string,
): RecentDatasetEntry[] {
  return entries.filter((entry) => entry.datasetIdentity !== datasetIdentity);
}

export function parseRecentDatasetStore(value: unknown): RecentDatasetStore | null {
  if (!isRecord(value)
    || value.schema !== RECENT_DATASETS_SCHEMA
    || value.schemaVersion !== RECENT_DATASETS_VERSION
    || !isContractTimestamp(value.savedAt)
    || !Array.isArray(value.entries)
    || value.entries.length > RECENT_DATASETS_LIMIT) return null;
  const entries = value.entries.map(parseEntry);
  if (entries.some((entry) => entry === null)) return null;
  const safeEntries = entries as RecentDatasetEntry[];
  if (new Set(safeEntries.map((entry) => entry.datasetIdentity)).size !== safeEntries.length) return null;
  return {
    schema: RECENT_DATASETS_SCHEMA,
    schemaVersion: RECENT_DATASETS_VERSION,
    savedAt: value.savedAt,
    entries: bounded(safeEntries),
  };
}

export function loadRecentDatasets(storage: ReadStorage): RecentDatasetEntry[] {
  try {
    const raw = storage.getItem(RECENT_DATASETS_STORAGE_KEY);
    if (!raw) return [];
    const store = parseRecentDatasetStore(JSON.parse(raw));
    return store ? [...store.entries] : [];
  } catch {
    return [];
  }
}

/** Returns false when browser persistence is unavailable or over quota. */
export function saveRecentDatasets(
  storage: WriteStorage,
  entries: readonly RecentDatasetEntry[],
  savedAt = new Date().toISOString(),
): boolean {
  if (!isContractTimestamp(savedAt)) return false;
  const normalized = bounded(entries);
  if (normalized.some((entry) => parseEntry(entry) === null)) return false;
  const store: RecentDatasetStore = {
    schema: RECENT_DATASETS_SCHEMA,
    schemaVersion: RECENT_DATASETS_VERSION,
    savedAt,
    entries: normalized,
  };
  try {
    storage.setItem(RECENT_DATASETS_STORAGE_KEY, JSON.stringify(store));
    return true;
  } catch {
    return false;
  }
}

export function clearRecentDatasets(storage: RemoveStorage): boolean {
  try {
    storage.removeItem(RECENT_DATASETS_STORAGE_KEY);
    return true;
  } catch {
    return false;
  }
}
