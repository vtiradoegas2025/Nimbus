import {
  DATASET_SOURCE_SCHEMA,
  DATASET_SOURCE_VERSION,
  type LocalDatasetSource,
} from "@/lib/dataset/datasetSources";

export const COMPARISON_SESSION_SCHEMA = "nimbus.studio.comparison-session" as const;
export const COMPARISON_SESSION_VERSION = 1 as const;
const STORAGE_KEY = "nimbus:comparison-session:v1";

export interface ComparisonSession {
  schema: typeof COMPARISON_SESSION_SCHEMA;
  schemaVersion: typeof COMPARISON_SESSION_VERSION;
  left: LocalDatasetSource;
  right: LocalDatasetSource;
  savedAt: string;
}

export function isLocalDatasetSource(value: unknown): value is LocalDatasetSource {
  if (!value || typeof value !== "object") return false;
  const source = value as Partial<LocalDatasetSource>;
  const locator = source.locator as Partial<LocalDatasetSource["locator"]> | undefined;
  const paths = locator?.paths;
  const locatorValid = locator?.kind === "local_path"
    ? paths == null
    : locator?.kind === "local_paths"
      && Array.isArray(paths)
      && paths.length >= 2
      && paths.every((path) => typeof path === "string" && path.length > 0)
      && locator.path === paths[0];
  const kindValid = source.kind === "nimbus_run"
    ? source.sourceModel === "nimbus" && source.nativeFormat === "nimbus_history" && locator?.kind === "local_path"
    : source.kind === "cm1_netcdf"
      && source.sourceModel === "cm1" && source.nativeFormat === "netcdf"
      && (locator?.kind === "local_path" || locator?.kind === "local_paths");
  return source.schema === DATASET_SOURCE_SCHEMA
    && source.schemaVersion === DATASET_SOURCE_VERSION
    && kindValid
    && source.readOnly === true
    && typeof source.displayName === "string" && source.displayName.trim().length > 0
    && locatorValid
    && typeof locator?.path === "string"
    && locator.path.length > 0;
}

export function createComparisonSession(left: LocalDatasetSource, right: LocalDatasetSource): ComparisonSession {
  return { schema: COMPARISON_SESSION_SCHEMA, schemaVersion: COMPARISON_SESSION_VERSION, left, right, savedAt: new Date().toISOString() };
}

export function parseComparisonSession(value: unknown): ComparisonSession | null {
  if (!value || typeof value !== "object") return null;
  const session = value as Partial<ComparisonSession>;
  return session.schema === COMPARISON_SESSION_SCHEMA
    && session.schemaVersion === COMPARISON_SESSION_VERSION
    && isLocalDatasetSource(session.left)
    && isLocalDatasetSource(session.right)
    && typeof session.savedAt === "string"
    ? session as ComparisonSession
    : null;
}

export function loadComparisonSession(storage: Pick<Storage, "getItem">): ComparisonSession | null {
  try {
    const value = storage.getItem(STORAGE_KEY);
    return value ? parseComparisonSession(JSON.parse(value)) : null;
  } catch { return null; }
}

export function saveComparisonSession(storage: Pick<Storage, "setItem">, session: ComparisonSession): void {
  // Guarded here rather than at each call site: persistence is best-effort,
  // and a quota failure surfacing through an unrelated caller's catch gets
  // reported as that caller's error (an import failure, say) instead.
  try {
    storage.setItem(STORAGE_KEY, JSON.stringify(session));
  } catch {
    // Storage full or blocked — the in-memory session stays authoritative.
  }
}

export function clearComparisonSession(storage: Pick<Storage, "removeItem">): void {
  storage.removeItem(STORAGE_KEY);
}
