/*
 * Editor buffers for the Python tab.
 * Files are canonical; localStorage is recovery-only (scratch text + dirty edits).
 * A clean file-backed buffer persists its path and is re-read on reopen.
 */

export const PYTHON_WORKSPACE_SCHEMA = "nimbus.studio.python-workspace" as const;
export const PYTHON_WORKSPACE_VERSION = 1 as const;
export const PYTHON_WORKSPACE_STORAGE_KEY = "nimbus-studio:python-workspace:v1" as const;
/** The single-buffer key this replaces. Read once, to migrate. */
export const LEGACY_SNIPPET_STORAGE_KEY = "nimbus-studio:python-snippet:last-code" as const;

/** Must match PaneContextValue's default so unsplit-view tabs use the pre-pane key without migration. */
export const PRIMARY_PANE_ID = "primary" as const;

// | is used as separator because encodeURIComponent escapes it to %7C, making the key injective.
const PANE_SEPARATOR = "|pane:";

/**
 * Recovery store key: scoped per project AND per pane.
 * Not on StudioProject: scratch text is up to 512 kB and is machine-local unsaved state, not analysis content.
 */
export function pythonWorkspaceStorageKey(
  projectId: string | null,
  paneId: string = PRIMARY_PANE_ID,
): string {
  const scoped = projectId === null
    ? PYTHON_WORKSPACE_STORAGE_KEY
    : `${PYTHON_WORKSPACE_STORAGE_KEY}:${encodeURIComponent(projectId)}`;
  if (paneId === PRIMARY_PANE_ID) return scoped;
  return `${scoped}${PANE_SEPARATOR}${encodeURIComponent(paneId)}`;
}

/** Inheritance source for a new workspace: project primary pane → unscoped store → null. */
export function pythonWorkspaceSeedKey(
  projectId: string | null,
  paneId: string = PRIMARY_PANE_ID,
): string | null {
  if (paneId !== PRIMARY_PANE_ID) return pythonWorkspaceStorageKey(projectId);
  if (projectId !== null) return PYTHON_WORKSPACE_STORAGE_KEY;
  return null;
}

/** Project id for a storage key, or null for unscoped keys. Strips the pane segment before decoding to avoid false non-matches. */
function projectIdForStorageKey(key: string): string | null {
  const prefix = `${PYTHON_WORKSPACE_STORAGE_KEY}:`;
  if (!key.startsWith(prefix)) return null;
  const tail = key.slice(prefix.length);
  const separator = tail.indexOf(PANE_SEPARATOR);
  const encoded = separator >= 0 ? tail.slice(0, separator) : tail;
  try {
    return decodeURIComponent(encoded);
  } catch {
    // Not a key this module wrote; leave it for whoever owns it.
    return null;
  }
}

/** Every store belonging to one project — one per pane it has been scripted in. */
export function pythonWorkspaceKeysForProject(
  storageKeys: readonly string[],
  projectId: string,
): string[] {
  return storageKeys.filter((key) => projectIdForStorageKey(key) === projectId);
}

/** Stores for deleted projects. Each may hold up to 512 kB; leaving them accumulates storage pressure. Unscoped stores are never orphaned. */
export function orphanedPythonWorkspaceKeys(
  storageKeys: readonly string[],
  liveProjectIds: ReadonlySet<string>,
): string[] {
  return storageKeys.filter((key) => {
    const projectId = projectIdForStorageKey(key);
    return projectId !== null && !liveProjectIds.has(projectId);
  });
}

export const MAX_BUFFERS = 12;

export class WorkspaceOperationOwner {
  private key: string;
  private generation = 0;
  constructor(key: string) { this.key = key; }
  select(key: string): void {
    if (key === this.key) return;
    this.key = key;
    this.generation++;
  }
  capture(): { key: string; generation: number } { return { key: this.key, generation: this.generation }; }
  current(owner: { key: string; generation: number }): boolean {
    return owner.key === this.key && owner.generation === this.generation;
  }
  retire(): void { this.generation++; }
}
/** Guard against a pathological paste filling the origin's storage quota. */
export const MAX_RECOVERED_BUFFER_BYTES = 512 * 1024;

/**
 * Buffers whose only current copy cannot fit in the recovery store.
 *
 * This is intentionally cheap enough to call while typing: string length is
 * already available, while encoding every open buffer to UTF-8 on each change
 * would turn a safety check into editor latency. The serializer uses the same
 * bound, so the warning and the write can never disagree.
 */
export function buffersOutsideRecoveryLimit(
  workspace: PythonWorkspace,
): readonly PythonBuffer[] {
  return workspace.buffers.filter((buffer) => {
    const needsRecoveryText = buffer.path === null || isBufferDirty(buffer);
    return needsRecoveryText
      && !buffer.needsReload
      && buffer.code.length > MAX_RECOVERED_BUFFER_BYTES;
  });
}

/** Persistent, user-facing warning for unsaved text omitted from recovery. */
export function workspaceRecoveryWarning(workspace: PythonWorkspace): string | null {
  const omitted = buffersOutsideRecoveryLimit(workspace);
  if (omitted.length === 0) return null;
  const names = omitted.map((buffer) => buffer.name).join(", ");
  return `${names} ${omitted.length === 1 ? "is" : "are"} too large for automatic recovery. `
    + "Save to disk before reloading or closing Studio.";
}

export interface PythonBuffer {
  readonly id: string;
  /** Display name: the file's basename, or a scratch label. */
  readonly name: string;
  /** Host path when file-backed; null for scratch. */
  readonly path: string | null;
  readonly code: string;
  /**
   * Contents as of the last read from or write to disk. Dirty is
   * `code !== savedCode`, so a scratch buffer starts clean against its starter
   * text and only reports unsaved once actually edited.
   */
  readonly savedCode: string;
  /**
   * Set when the buffer was restored by path alone and its text has not been
   * re-read yet. The component loads it on activation; until then `code` is
   * empty and must not be written back to disk.
   */
  readonly needsReload: boolean;
}

export interface PythonWorkspace {
  readonly buffers: readonly PythonBuffer[];
  readonly activeId: string;
}

export function isBufferDirty(buffer: PythonBuffer): boolean {
  return !buffer.needsReload && buffer.code !== buffer.savedCode;
}

export function workspaceHasUnsaved(workspace: PythonWorkspace): boolean {
  return workspace.buffers.some(isBufferDirty);
}

export function activeBuffer(workspace: PythonWorkspace): PythonBuffer {
  return workspace.buffers.find((buffer) => buffer.id === workspace.activeId)
    ?? workspace.buffers[0];
}

/** Final path segment, tolerating either separator. */
export function fileName(path: string): string {
  const segments = path.split(/[/\\]/);
  return segments[segments.length - 1] || path;
}

/** Lowest unused `buf_N`, so ids are stable and testable without a clock. */
export function nextBufferId(existing: readonly PythonBuffer[]): string {
  const taken = new Set(existing.map((buffer) => buffer.id));
  for (let index = 1; ; index += 1) {
    const candidate = `buf_${index}`;
    if (!taken.has(candidate)) return candidate;
  }
}

/** Scratch labels count up so two unsaved buffers stay distinguishable. */
export function nextScratchName(existing: readonly PythonBuffer[]): string {
  const taken = new Set(existing.map((buffer) => buffer.name));
  if (!taken.has("Scratch")) return "Scratch";
  for (let index = 2; ; index += 1) {
    const candidate = `Scratch ${index}`;
    if (!taken.has(candidate)) return candidate;
  }
}

export function createScratchBuffer(
  existing: readonly PythonBuffer[],
  starterCode: string,
): PythonBuffer {
  return {
    id: nextBufferId(existing),
    name: nextScratchName(existing),
    path: null,
    code: starterCode,
    savedCode: starterCode,
    needsReload: false,
  };
}

export function createWorkspace(starterCode: string): PythonWorkspace {
  const scratch = createScratchBuffer([], starterCode);
  return { buffers: [scratch], activeId: scratch.id };
}

export function setBufferCode(
  workspace: PythonWorkspace,
  id: string,
  code: string,
): PythonWorkspace {
  return {
    ...workspace,
    buffers: workspace.buffers.map((buffer) =>
      buffer.id === id ? { ...buffer, code, needsReload: false } : buffer
    ),
  };
}

/** Record a successful disk read: text is now known and matches disk. */
export function markBufferLoaded(
  workspace: PythonWorkspace,
  id: string,
  code: string,
): PythonWorkspace {
  return {
    ...workspace,
    buffers: workspace.buffers.map((buffer) =>
      buffer.id === id ? { ...buffer, code, savedCode: code, needsReload: false } : buffer
    ),
  };
}

/** Record exact written text; edits made during the write remain unsaved. */
export function markBufferSaved(
  workspace: PythonWorkspace,
  id: string,
  path: string,
  writtenCode?: string,
): PythonWorkspace {
  return {
    ...workspace,
    buffers: workspace.buffers.map((buffer) =>
      buffer.id === id
        ? { ...buffer, path, name: fileName(path), savedCode: writtenCode ?? buffer.code, needsReload: false }
        : buffer
    ),
  };
}

export function activateBuffer(workspace: PythonWorkspace, id: string): PythonWorkspace {
  return workspace.buffers.some((buffer) => buffer.id === id)
    ? { ...workspace, activeId: id }
    : workspace;
}

export interface OpenPathResult {
  readonly workspace: PythonWorkspace;
  /** False when an existing tab for this path was focused instead. */
  readonly opened: boolean;
  readonly bufferId: string;
}

/** Open a file, reusing an existing tab for the same path. Clean buffers refresh; dirty buffers keep their edits. */
export function openPathInWorkspace(
  workspace: PythonWorkspace,
  path: string,
  code: string,
): OpenPathResult {
  const existing = workspace.buffers.find((buffer) => buffer.path === path);
  if (existing) {
    const refreshed = isBufferDirty(existing)
      ? existing
      : { ...existing, code, savedCode: code, needsReload: false };
    return {
      workspace: {
        buffers: workspace.buffers.map((buffer) => (buffer.id === existing.id ? refreshed : buffer)),
        activeId: existing.id,
      },
      opened: false,
      bufferId: existing.id,
    };
  }
  const buffer: PythonBuffer = {
    id: nextBufferId(workspace.buffers),
    name: fileName(path),
    path,
    code,
    savedCode: code,
    needsReload: false,
  };
  return {
    workspace: { buffers: [...workspace.buffers, buffer], activeId: buffer.id },
    opened: true,
    bufferId: buffer.id,
  };
}

/** Close a buffer. Last-closed becomes a fresh scratch; active-tab close focuses the neighbour. */
export function closeBuffer(
  workspace: PythonWorkspace,
  id: string,
  starterCode: string,
): PythonWorkspace {
  const index = workspace.buffers.findIndex((buffer) => buffer.id === id);
  if (index < 0) return workspace;
  const remaining = workspace.buffers.filter((buffer) => buffer.id !== id);
  if (remaining.length === 0) return createWorkspace(starterCode);
  const activeId = workspace.activeId === id
    ? remaining[Math.min(index, remaining.length - 1)].id
    : workspace.activeId;
  return { buffers: remaining, activeId };
}

export function canAddBuffer(workspace: PythonWorkspace): boolean {
  return workspace.buffers.length < MAX_BUFFERS;
}

export function addScratchBuffer(
  workspace: PythonWorkspace,
  starterCode: string,
): PythonWorkspace {
  if (!canAddBuffer(workspace)) return workspace;
  const buffer = createScratchBuffer(workspace.buffers, starterCode);
  return { buffers: [...workspace.buffers, buffer], activeId: buffer.id };
}

/* ---------- Recovery storage ---------- */

interface PersistedBuffer {
  id: string;
  name: string;
  path: string | null;
  /** Omitted when the text is already on disk and unmodified. */
  code?: string;
}

/** Serialise for recovery. Clean file-backed buffers contribute path only; scratch/dirty buffers include their text. */
export function serializeWorkspace(workspace: PythonWorkspace): string {
  const buffers: PersistedBuffer[] = workspace.buffers.map((buffer) => {
    const record: PersistedBuffer = { id: buffer.id, name: buffer.name, path: buffer.path };
    const needsText = buffer.path === null || isBufferDirty(buffer);
    if (needsText && !buffer.needsReload && buffer.code.length <= MAX_RECOVERED_BUFFER_BYTES) {
      record.code = buffer.code;
    }
    return record;
  });
  return JSON.stringify({
    schema: PYTHON_WORKSPACE_SCHEMA,
    schemaVersion: PYTHON_WORKSPACE_VERSION,
    activeId: workspace.activeId,
    buffers,
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Restore a workspace. Malformed input returns a fresh workspace; partial recovery risks showing wrong text under a real filename. */
export function parseWorkspace(raw: string | null, starterCode: string): PythonWorkspace {
  if (!raw) return createWorkspace(starterCode);
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return createWorkspace(starterCode);
  }
  if (!isRecord(value)
    || value.schema !== PYTHON_WORKSPACE_SCHEMA
    || value.schemaVersion !== PYTHON_WORKSPACE_VERSION
    || !Array.isArray(value.buffers)) {
    return createWorkspace(starterCode);
  }

  const buffers: PythonBuffer[] = [];
  const seenIds = new Set<string>();
  for (const entry of value.buffers.slice(0, MAX_BUFFERS)) {
    if (!isRecord(entry)) continue;
    const { id, name, path, code } = entry;
    if (typeof id !== "string" || !id || seenIds.has(id)) continue;
    if (typeof name !== "string" || !name) continue;
    if (path !== null && typeof path !== "string") continue;
    const hasCode = typeof code === "string" && code.length <= MAX_RECOVERED_BUFFER_BYTES;
    // No text and no path is an empty shell that could never be populated.
    if (!hasCode && path === null) continue;
    seenIds.add(id);
    buffers.push({
      id,
      name,
      path: path ?? null,
      code: hasCode ? code as string : "",
      // Persisted text was, by construction, either scratch or an unsaved
      // edit; either way it does not match disk, so it restores dirty.
      savedCode: hasCode && path === null ? code as string : "",
      needsReload: !hasCode,
    });
  }
  if (buffers.length === 0) return createWorkspace(starterCode);

  const activeId = typeof value.activeId === "string" && seenIds.has(value.activeId)
    ? value.activeId
    : buffers[0].id;
  return { buffers, activeId };
}

/**
 * Seed a new project's tab strip from the unscoped store.
 * Only file-backed buffers carry over (as paths). Scratch text stays in its original store to avoid forks.
 */
export function seedProjectWorkspace(
  unscopedRaw: string | null,
  starterCode: string,
): PythonWorkspace {
  const source = parseWorkspace(unscopedRaw, starterCode);
  const buffers: PythonBuffer[] = source.buffers
    .filter((buffer): buffer is PythonBuffer & { path: string } => buffer.path !== null)
    .map((buffer) => ({
      id: buffer.id,
      name: buffer.name,
      path: buffer.path,
      code: "",
      savedCode: "",
      needsReload: true,
    }));
  if (buffers.length === 0) return createWorkspace(starterCode);
  const activeId = buffers.some((buffer) => buffer.id === source.activeId)
    ? source.activeId
    : buffers[0].id;
  return { buffers, activeId };
}

/** Migrate the pre-workspace single-buffer store. Text becomes a scratch buffer rather than being discarded. */
export function migrateLegacySnippet(
  legacyCode: string | null,
  starterCode: string,
): PythonWorkspace {
  if (!legacyCode) return createWorkspace(starterCode);
  const workspace = createWorkspace(starterCode);
  const [scratch] = workspace.buffers;
  return {
    buffers: [{ ...scratch, code: legacyCode, savedCode: legacyCode }],
    activeId: scratch.id,
  };
}
