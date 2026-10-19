import { fnv1a64Text } from "@/lib/cm1/cm1GuidedHandoff";

export const CONFIG_DRAFT_STORAGE_KEY = "nimbus:studio:config-drafts:v1" as const;
export const CONFIG_DRAFT_LIMIT = 16 as const;
export const CONFIG_DRAFT_DOCUMENT_MAX_CHARS = 512_000 as const;
export const CONFIG_DRAFT_STORE_MAX_BYTES = 1_500_000 as const;

export type ConfigDraftViewMode = "form" | "raw";

export interface ConfigDraftSaveEvidence {
  path: string;
  documentFingerprint: string;
  savedAt: string;
}

export interface ConfigDraft {
  schema: "nimbus.studio.config-draft";
  schemaVersion: 1;
  id: string;
  parentId: string | null;
  caseName: string;
  document: string;
  viewMode: ConfigDraftViewMode;
  selectedTemplate: string;
  sourcePath: string | null;
  createdAt: string;
  updatedAt: string;
  saved: ConfigDraftSaveEvidence | null;
}

export interface ConfigDraftStore {
  schema: "nimbus.studio.config-draft-store";
  schemaVersion: 1;
  activeId: string;
  drafts: ConfigDraft[];
}

type ReadStorage = Pick<Storage, "getItem">;
type WriteStorage = Pick<Storage, "setItem">;

export type ConfigDraftLoadResult =
  | { kind: "empty" }
  | { kind: "ready"; store: ConfigDraftStore }
  | { kind: "unreadable"; message: string };

const DRAFT_ID = /^draft_[a-z0-9][a-z0-9-]{7,79}$/;

function canonicalTime(value: unknown): value is string {
  return typeof value === "string"
    && Number.isFinite(Date.parse(value))
    && new Date(value).toISOString() === value;
}

function boundedText(value: unknown, maximum: number, allowEmpty = false): value is string {
  return typeof value === "string"
    && !value.includes("\0")
    && value.length <= maximum
    && (allowEmpty || value.length > 0);
}

function validSaved(value: unknown): value is ConfigDraftSaveEvidence {
  if (!value || typeof value !== "object") return false;
  const record = value as Record<string, unknown>;
  return Object.keys(record).length === 3
    && boundedText(record.path, 4_096)
    && typeof record.documentFingerprint === "string"
    && /^fnv1a64:[a-f0-9]{16}$/.test(record.documentFingerprint)
    && canonicalTime(record.savedAt);
}

export function parseConfigDraft(value: unknown): ConfigDraft | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  const keys = [
    "schema", "schemaVersion", "id", "parentId", "caseName", "document",
    "viewMode", "selectedTemplate", "sourcePath", "createdAt", "updatedAt", "saved",
  ];
  if (Object.keys(record).length !== keys.length || keys.some((key) => !(key in record))) return null;
  if (record.schema !== "nimbus.studio.config-draft" || record.schemaVersion !== 1) return null;
  if (typeof record.id !== "string" || !DRAFT_ID.test(record.id)) return null;
  if (record.parentId !== null && (typeof record.parentId !== "string" || !DRAFT_ID.test(record.parentId))) return null;
  if (!boundedText(record.caseName, 96, true)
    || !boundedText(record.document, CONFIG_DRAFT_DOCUMENT_MAX_CHARS, true)
    || (record.viewMode !== "form" && record.viewMode !== "raw")
    || !boundedText(record.selectedTemplate, 512, true)
    || (record.sourcePath !== null && !boundedText(record.sourcePath, 4_096))
    || !canonicalTime(record.createdAt)
    || !canonicalTime(record.updatedAt)
    || Date.parse(record.updatedAt) < Date.parse(record.createdAt)
    || (record.saved !== null && !validSaved(record.saved))) return null;
  return {
    schema: "nimbus.studio.config-draft",
    schemaVersion: 1,
    id: record.id,
    parentId: record.parentId,
    caseName: record.caseName,
    document: record.document,
    viewMode: record.viewMode,
    selectedTemplate: record.selectedTemplate,
    sourcePath: record.sourcePath,
    createdAt: record.createdAt,
    updatedAt: record.updatedAt,
    saved: record.saved,
  };
}

export function parseConfigDraftStore(value: unknown): ConfigDraftStore | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  if (Object.keys(record).length !== 4
    || record.schema !== "nimbus.studio.config-draft-store"
    || record.schemaVersion !== 1
    || typeof record.activeId !== "string"
    || !Array.isArray(record.drafts)
    || record.drafts.length < 1
    || record.drafts.length > CONFIG_DRAFT_LIMIT) return null;
  const drafts = record.drafts.map(parseConfigDraft);
  if (drafts.some((draft) => draft == null)) return null;
  const canonical = drafts as ConfigDraft[];
  if (new Set(canonical.map(({ id }) => id)).size !== canonical.length
    || !canonical.some(({ id }) => id === record.activeId)
    || canonical.some(({ parentId }) => parentId != null && !canonical.some(({ id }) => id === parentId))) return null;
  return {
    schema: "nimbus.studio.config-draft-store",
    schemaVersion: 1,
    activeId: record.activeId,
    drafts: canonical.sort((left, right) => Date.parse(right.updatedAt) - Date.parse(left.updatedAt)),
  };
}

export function inspectConfigDraftStore(storage?: ReadStorage): ConfigDraftLoadResult {
  if (!storage) return { kind: "empty" };
  try {
    const serialized = storage.getItem(CONFIG_DRAFT_STORAGE_KEY);
    if (!serialized) return { kind: "empty" };
    if (new TextEncoder().encode(serialized).byteLength > CONFIG_DRAFT_STORE_MAX_BYTES) {
      return {
        kind: "unreadable",
        message: "Stored configuration recovery data exceeds the 1.5 MB safety limit.",
      };
    }
    const store = parseConfigDraftStore(JSON.parse(serialized));
    return store
      ? { kind: "ready", store }
      : {
          kind: "unreadable",
          message: "Stored configuration recovery data does not satisfy the versioned draft contract.",
        };
  } catch (reason) {
    return {
      kind: "unreadable",
      message: reason instanceof Error
        ? `Configuration recovery storage could not be read: ${reason.message}`
        : "Configuration recovery storage could not be read.",
    };
  }
}

export function loadConfigDraftStore(storage?: ReadStorage): ConfigDraftStore | null {
  const result = inspectConfigDraftStore(storage);
  return result.kind === "ready" ? result.store : null;
}

export function saveConfigDraftStore(storage: WriteStorage | undefined, store: ConfigDraftStore): void {
  if (!storage) return;
  const parsed = parseConfigDraftStore(store);
  if (!parsed) throw new Error("Configuration drafts do not satisfy the versioned local contract.");
  const serialized = JSON.stringify(parsed);
  if (new TextEncoder().encode(serialized).byteLength > CONFIG_DRAFT_STORE_MAX_BYTES) {
    throw new Error("Configuration drafts exceed the 1.5 MB local safety limit. Save or remove an older branch.");
  }
  storage.setItem(CONFIG_DRAFT_STORAGE_KEY, serialized);
}

function generatedId(now: string): string {
  const suffix = globalThis.crypto?.randomUUID?.().toLowerCase().replace(/[^a-z0-9]/g, "").slice(0, 16)
    ?? Math.random().toString(36).slice(2, 18).padEnd(8, "0");
  return `draft_${Date.parse(now).toString(36)}-${suffix}`.slice(0, 80);
}

export function createConfigDraft(input: {
  caseName: string;
  document: string;
  viewMode?: ConfigDraftViewMode;
  selectedTemplate?: string;
  sourcePath?: string | null;
  parentId?: string | null;
  savedPath?: string | null;
}, options: { id?: string; now?: string } = {}): ConfigDraft {
  const now = options.now ?? new Date().toISOString();
  const candidate: ConfigDraft = {
    schema: "nimbus.studio.config-draft",
    schemaVersion: 1,
    id: options.id ?? generatedId(now),
    parentId: input.parentId ?? null,
    caseName: input.caseName,
    document: input.document,
    viewMode: input.viewMode ?? "form",
    selectedTemplate: input.selectedTemplate ?? "",
    sourcePath: input.sourcePath ?? null,
    createdAt: now,
    updatedAt: now,
    saved: input.savedPath ? {
      path: input.savedPath,
      documentFingerprint: fnv1a64Text(input.document),
      savedAt: now,
    } : null,
  };
  const parsed = parseConfigDraft(candidate);
  if (!parsed) throw new Error("Configuration draft input is invalid or exceeds its safety bounds.");
  return parsed;
}

export function configDraftStore(draft: ConfigDraft): ConfigDraftStore {
  const parsed = parseConfigDraft(draft);
  if (!parsed) throw new Error("Cannot create a draft store from an invalid draft.");
  return { schema: "nimbus.studio.config-draft-store", schemaVersion: 1, activeId: parsed.id, drafts: [parsed] };
}

export function updateConfigDraft(
  store: ConfigDraftStore,
  id: string,
  update: Partial<Pick<ConfigDraft, "caseName" | "document" | "viewMode" | "selectedTemplate" | "sourcePath" | "saved">>,
  now = new Date().toISOString(),
): ConfigDraftStore {
  const current = store.drafts.find((draft) => draft.id === id);
  if (!current) throw new Error("Configuration draft was not found.");
  const next = parseConfigDraft({ ...current, ...update, updatedAt: now });
  if (!next) throw new Error("Configuration draft update is invalid or exceeds its safety bounds.");
  return parseConfigDraftStore({
    ...store,
    activeId: id,
    drafts: [next, ...store.drafts.filter((draft) => draft.id !== id)],
  })!;
}

export function branchConfigDraft(
  store: ConfigDraftStore,
  sourceId: string,
  caseName: string,
  options: { id?: string; now?: string } = {},
): ConfigDraftStore {
  if (store.drafts.length >= CONFIG_DRAFT_LIMIT) throw new Error(`Studio supports at most ${CONFIG_DRAFT_LIMIT} configuration drafts.`);
  const source = store.drafts.find((draft) => draft.id === sourceId);
  if (!source) throw new Error("Configuration draft to branch was not found.");
  const branch = createConfigDraft({
    caseName,
    document: source.document,
    viewMode: source.viewMode,
    selectedTemplate: source.selectedTemplate,
    sourcePath: null,
    parentId: source.id,
  }, options);
  return { ...store, activeId: branch.id, drafts: [branch, ...store.drafts] };
}

export function activateConfigDraft(store: ConfigDraftStore, id: string): ConfigDraftStore {
  if (!store.drafts.some((draft) => draft.id === id)) throw new Error("Configuration draft to activate was not found.");
  return { ...store, activeId: id };
}

export function deleteConfigDraft(store: ConfigDraftStore, id: string): ConfigDraftStore {
  if (store.drafts.length === 1) throw new Error("Keep at least one configuration draft.");
  const drafts = store.drafts
    .filter((draft) => draft.id !== id)
    .map((draft) => draft.parentId === id ? { ...draft, parentId: null } : draft);
  if (drafts.length === store.drafts.length) throw new Error("Configuration draft to remove was not found.");
  return { ...store, activeId: store.activeId === id ? drafts[0].id : store.activeId, drafts };
}

export function markConfigDraftSaved(
  store: ConfigDraftStore,
  id: string,
  document: string,
  path: string,
  now = new Date().toISOString(),
): ConfigDraftStore {
  return updateConfigDraft(store, id, {
    document,
    sourcePath: path,
    saved: { path, documentFingerprint: fnv1a64Text(document), savedAt: now },
  }, now);
}

export function configDraftIsDirty(draft: ConfigDraft, document = draft.document): boolean {
  return draft.saved == null || draft.saved.documentFingerprint !== fnv1a64Text(document);
}
