import { qualifyKey, SCHEMA_SECTIONS, type FieldSchema } from "@/lib/config/configSchema";
import type {
  Cm1CompatibilityEntry,
  Cm1CompatibilityReport,
  Cm1CompatibilityStatus,
} from "./cm1Compatibility";
import { datasetSourceIdentity, type LocalDatasetSource } from "@/lib/dataset/datasetSources";

export const CM1_DECISION_WORKBOOK_SCHEMA = "nimbus.studio.cm1-decision-workbook" as const;
export const CM1_DECISION_WORKBOOK_VERSION = 2 as const;
export const CM1_DECISION_WORKBOOK_MAX_JSON_BYTES = 2_000_000;
export const CM1_DECISION_NOTE_MAX_CHARS = 1_000;
export const CM1_MANUAL_VALUE_MAX_CHARS = 512;
const MAX_ITEMS = 5_000;
const MAX_SOURCE_IDENTITY_CHARS = 256_000;
const MAX_SNAPSHOT_STRING_CHARS = 4_096;

export type Cm1DecisionKind =
  | "keep_nimbus_default"
  | "manual_resolution"
  | "acknowledge_difference"
  | "omit_unsupported";

export interface Cm1MappingDecision {
  kind: Cm1DecisionKind;
  manualValue: string | null;
  recordedDefault: string | number | boolean | null;
  note: string | null;
  decidedAt: string;
}

export interface Cm1DecisionWorkbookItem {
  id: string;
  group: string;
  key: string;
  rawValue: string;
  status: Exclude<Cm1CompatibilityStatus, "ready">;
  target: string | null;
  reason: string;
  targetDescriptor: Cm1TargetDescriptorSnapshot | null;
  decision: Cm1MappingDecision | null;
}

export interface Cm1TargetDescriptorSnapshot {
  fingerprint: string;
  kind: FieldSchema["kind"];
  defaultValue: string | number | boolean;
  minimum: number | null;
  maximum: number | null;
  locked: boolean;
  enumOptions: Array<{ value: string; disabled: boolean }>;
}

export interface Cm1DecisionWorkbook {
  schema: typeof CM1_DECISION_WORKBOOK_SCHEMA;
  schemaVersion: typeof CM1_DECISION_WORKBOOK_VERSION;
  runnable: false;
  scientificEquivalenceClaimed: false;
  source: {
    datasetIdentity: string;
    mappingFingerprint: string;
    descriptorFingerprint: string;
    displayName: string;
  };
  items: Cm1DecisionWorkbookItem[];
  createdAt: string;
  updatedAt: string;
}

export interface Cm1DecisionProgress {
  complete: number;
  remaining: number;
  total: number;
}

type WorkbookStorageReader = Pick<Storage, "getItem">;
type WorkbookStorageWriter = Pick<Storage, "setItem">;

const fieldsByPath = new Map<string, FieldSchema>();
for (const section of SCHEMA_SECTIONS) {
  for (const field of section.fields) fieldsByPath.set(qualifyKey(section, field), field);
}

function boundedString(value: unknown, max: number, allowEmpty = false): value is string {
  return typeof value === "string" && value.length <= max && (allowEmpty || value.trim().length > 0);
}

function fnv1a(input: string): string {
  let hash = 0x811c9dc5;
  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

function mappingSnapshot(items: readonly Cm1DecisionWorkbookItem[]): string {
  return JSON.stringify(items.map(({ id, group, key, rawValue, status, target, reason }) => [
    id, group, key, rawValue, status, target, reason,
  ]));
}

function fingerprint(items: readonly Cm1DecisionWorkbookItem[]): string {
  return `fnv1a32:${fnv1a(mappingSnapshot(items))}:${items.length}`;
}

function descriptorSnapshot(target: string | null): Cm1TargetDescriptorSnapshot | null {
  const field = target ? fieldsByPath.get(target) : undefined;
  if (!field) return null;
  const content = {
    kind: field.kind,
    defaultValue: field.default,
    minimum: field.kind === "int" || field.kind === "float" ? field.min ?? null : null,
    maximum: field.kind === "int" || field.kind === "float" ? field.max ?? null : null,
    locked: field.locked === true,
    enumOptions: field.kind === "enum"
      ? field.options.map(({ value, disabled }) => ({ value, disabled: disabled === true }))
      : [],
  };
  return { fingerprint: `fnv1a32:${fnv1a(JSON.stringify(content))}`, ...content };
}

function descriptorFingerprint(items: readonly Cm1DecisionWorkbookItem[]): string {
  return `fnv1a32:${fnv1a(JSON.stringify(items.map(({ id, targetDescriptor }) => [id, targetDescriptor])))}`;
}

function descriptorWorkbookSnapshot(items: readonly Cm1DecisionWorkbookItem[]): string {
  return JSON.stringify(items.map(({ id, targetDescriptor }) => [id, targetDescriptor]));
}

function toWorkbookItem(entry: Cm1CompatibilityEntry): Cm1DecisionWorkbookItem | null {
  if (entry.status === "ready") return null;
  return { ...entry, status: entry.status, targetDescriptor: descriptorSnapshot(entry.target), decision: null };
}

export function createCm1DecisionWorkbook(
  dataset: LocalDatasetSource,
  report: Cm1CompatibilityReport,
  now = new Date().toISOString(),
): Cm1DecisionWorkbook {
  if (dataset.kind !== "cm1_netcdf") throw new Error("A CM1 decision workbook requires a CM1 source.");
  const datasetIdentity = datasetSourceIdentity(dataset);
  const items = report.entries.flatMap((entry) => {
    const item = toWorkbookItem(entry);
    return item ? [item] : [];
  });
  if (items.length > MAX_ITEMS) throw new Error(`A CM1 decision workbook supports at most ${MAX_ITEMS} mapping items.`);
  if (!boundedString(datasetIdentity, MAX_SOURCE_IDENTITY_CHARS)) {
    throw new Error("The CM1 source identity is too large for a bounded decision workbook.");
  }
  if (items.some((item) => [item.id, item.group, item.key, item.rawValue, item.reason]
    .some((value) => !boundedString(value, MAX_SNAPSHOT_STRING_CHARS, true)))) {
    throw new Error("One or more CM1 mapping fields exceed the bounded workbook contract.");
  }
  const workbook: Cm1DecisionWorkbook = {
    schema: CM1_DECISION_WORKBOOK_SCHEMA,
    schemaVersion: CM1_DECISION_WORKBOOK_VERSION,
    runnable: false,
    scientificEquivalenceClaimed: false,
    source: {
      datasetIdentity,
      mappingFingerprint: fingerprint(items),
      descriptorFingerprint: descriptorFingerprint(items),
      displayName: dataset.displayName,
    },
    items,
    createdAt: now,
    updatedAt: now,
  };
  if (serializeCm1DecisionWorkbook(workbook).length > CM1_DECISION_WORKBOOK_MAX_JSON_BYTES) {
    throw new Error("The CM1 decision workbook exceeds the portable JSON size limit.");
  }
  return workbook;
}

function decisionAllowed(item: Cm1DecisionWorkbookItem, kind: Cm1DecisionKind): boolean {
  if (item.status === "needs_decision") {
    const manualField = item.targetDescriptor;
    return kind === "keep_nimbus_default"
      || (kind === "manual_resolution" && manualField != null && manualField.locked !== true);
  }
  if (item.status === "different") return kind === "acknowledge_difference";
  return kind === "omit_unsupported";
}

export function cm1DecisionOptions(item: Cm1DecisionWorkbookItem): Cm1DecisionKind[] {
  const candidates: Cm1DecisionKind[] = [
    "keep_nimbus_default",
    "manual_resolution",
    "acknowledge_difference",
    "omit_unsupported",
  ];
  return candidates.filter((kind) => decisionAllowed(item, kind));
}

export function validateCm1ManualValue(
  item: Cm1DecisionWorkbookItem,
  rawValue: string,
): { valid: true; canonicalValue: string } | { valid: false; message: string } {
  if (!decisionAllowed(item, "manual_resolution") || !item.target) {
    return { valid: false, message: "This mapping has no supported manual Nimbus target." };
  }
  const value = rawValue.trim();
  if (!boundedString(value, CM1_MANUAL_VALUE_MAX_CHARS)) {
    return { valid: false, message: `Enter a value no longer than ${CM1_MANUAL_VALUE_MAX_CHARS} characters.` };
  }
  const field = item.targetDescriptor!;
  if (field.kind === "int" || field.kind === "float") {
    const normalized = value.replace(/[dD]/g, "e");
    const numberValue = Number(normalized);
    if (!Number.isFinite(numberValue)) return { valid: false, message: "Nimbus requires one finite numeric scalar." };
    if (field.kind === "int" && !Number.isInteger(numberValue)) return { valid: false, message: "Nimbus requires an integer value." };
    if (field.minimum != null && numberValue < field.minimum) return { valid: false, message: `Value is below the current minimum ${field.minimum}.` };
    if (field.maximum != null && numberValue > field.maximum) return { valid: false, message: `Value exceeds the current maximum ${field.maximum}.` };
    return { valid: true, canonicalValue: String(numberValue) };
  }
  if (field.kind === "bool") {
    const normalized = value.toLowerCase();
    return normalized === "true" || normalized === "false"
      ? { valid: true, canonicalValue: normalized }
      : { valid: false, message: "Nimbus requires true or false." };
  }
  if (field.kind === "enum") {
    const option = field.enumOptions.find((candidate) => candidate.value === value);
    if (!option) return { valid: false, message: "Choose a token offered by the current Nimbus field descriptor." };
    if (option.disabled) return { valid: false, message: "That Nimbus option is currently disabled." };
    return { valid: true, canonicalValue: option.value };
  }
  return { valid: true, canonicalValue: value };
}

export function decideCm1MappingItem(
  workbook: Cm1DecisionWorkbook,
  entryId: string,
  input: { kind: Cm1DecisionKind; manualValue?: string; note?: string },
  now = new Date().toISOString(),
): Cm1DecisionWorkbook {
  const index = workbook.items.findIndex((item) => item.id === entryId);
  if (index < 0) throw new Error("That mapping item does not belong to this source-bound workbook.");
  const item = workbook.items[index];
  if (!decisionAllowed(item, input.kind)) throw new Error("That decision is not valid for this mapping disposition.");
  const note = input.note?.trim() || null;
  if (note && note.length > CM1_DECISION_NOTE_MAX_CHARS) {
    throw new Error(`Review notes are limited to ${CM1_DECISION_NOTE_MAX_CHARS} characters.`);
  }
  let manualValue: string | null = null;
  if (input.kind === "manual_resolution") {
    const validation = validateCm1ManualValue(item, input.manualValue ?? "");
    if (!validation.valid) throw new Error(validation.message);
    manualValue = validation.canonicalValue;
  }
  const items = [...workbook.items];
  const recordedDefault = input.kind === "keep_nimbus_default"
    ? item.targetDescriptor?.defaultValue ?? null
    : null;
  items[index] = { ...item, decision: { kind: input.kind, manualValue, recordedDefault, note, decidedAt: now } };
  return { ...workbook, items, updatedAt: now };
}

export function clearCm1MappingDecision(
  workbook: Cm1DecisionWorkbook,
  entryId: string,
  now = new Date().toISOString(),
): Cm1DecisionWorkbook {
  const index = workbook.items.findIndex((item) => item.id === entryId);
  if (index < 0) throw new Error("That mapping item does not belong to this source-bound workbook.");
  const items = [...workbook.items];
  items[index] = { ...items[index], decision: null };
  return { ...workbook, items, updatedAt: now };
}

export function cm1DecisionProgress(workbook: Cm1DecisionWorkbook): Cm1DecisionProgress {
  const complete = workbook.items.filter((item) => item.decision != null).length;
  return { complete, remaining: workbook.items.length - complete, total: workbook.items.length };
}

function parseDecision(value: unknown, item: Cm1DecisionWorkbookItem): Cm1MappingDecision | null | undefined {
  if (value === null) return null;
  if (!value || typeof value !== "object") return undefined;
  const decision = value as Partial<Cm1MappingDecision>;
  if (!decision.kind || !decisionAllowed(item, decision.kind)) return undefined;
  if (!boundedString(decision.decidedAt, 64)) return undefined;
  if (!(decision.note === null || boundedString(decision.note, CM1_DECISION_NOTE_MAX_CHARS))) return undefined;
  if (decision.kind === "manual_resolution") {
    if (typeof decision.manualValue !== "string") return undefined;
    const validation = validateCm1ManualValue(item, decision.manualValue);
    if (!validation.valid || validation.canonicalValue !== decision.manualValue) return undefined;
  } else if (decision.manualValue !== null) return undefined;
  const expectedDefault = decision.kind === "keep_nimbus_default"
    ? item.targetDescriptor?.defaultValue ?? null
    : null;
  if (decision.recordedDefault !== expectedDefault) return undefined;
  return decision as Cm1MappingDecision;
}

function parseTargetDescriptor(value: unknown): Cm1TargetDescriptorSnapshot | null | undefined {
  if (value === null) return null;
  if (!value || typeof value !== "object") return undefined;
  const descriptor = value as Partial<Cm1TargetDescriptorSnapshot>;
  if ((descriptor.kind !== "int" && descriptor.kind !== "float" && descriptor.kind !== "bool" && descriptor.kind !== "string" && descriptor.kind !== "enum")
    || (typeof descriptor.defaultValue !== "string" && typeof descriptor.defaultValue !== "number" && typeof descriptor.defaultValue !== "boolean")
    || !(descriptor.minimum === null || typeof descriptor.minimum === "number")
    || !(descriptor.maximum === null || typeof descriptor.maximum === "number")
    || typeof descriptor.locked !== "boolean"
    || !Array.isArray(descriptor.enumOptions)
    || descriptor.enumOptions.length > 1_000
    || descriptor.enumOptions.some((option) => !option || !boundedString(option.value, 512) || typeof option.disabled !== "boolean")
    || !boundedString(descriptor.fingerprint, 64)) return undefined;
  const content = {
    kind: descriptor.kind,
    defaultValue: descriptor.defaultValue,
    minimum: descriptor.minimum,
    maximum: descriptor.maximum,
    locked: descriptor.locked,
    enumOptions: descriptor.enumOptions,
  };
  if (descriptor.fingerprint !== `fnv1a32:${fnv1a(JSON.stringify(content))}`) return undefined;
  return descriptor as Cm1TargetDescriptorSnapshot;
}

export function parseCm1DecisionWorkbook(value: unknown): Cm1DecisionWorkbook | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Partial<Cm1DecisionWorkbook>;
  const source = record.source as Partial<Cm1DecisionWorkbook["source"]> | undefined;
  if (record.schema !== CM1_DECISION_WORKBOOK_SCHEMA
    || record.schemaVersion !== CM1_DECISION_WORKBOOK_VERSION
    || record.runnable !== false
    || record.scientificEquivalenceClaimed !== false
    || !boundedString(source?.datasetIdentity, MAX_SOURCE_IDENTITY_CHARS)
    || !boundedString(source.mappingFingerprint, 64)
    || !boundedString(source.descriptorFingerprint, 64)
    || !boundedString(source.displayName, 512)
    || !Array.isArray(record.items)
    || record.items.length > MAX_ITEMS
    || !boundedString(record.createdAt, 64)
    || !boundedString(record.updatedAt, 64)) return null;
  const ids = new Set<string>();
  const items: Cm1DecisionWorkbookItem[] = [];
  for (const candidate of record.items) {
    if (!candidate || typeof candidate !== "object") return null;
    const item = candidate as Partial<Cm1DecisionWorkbookItem>;
    if (!boundedString(item.id, MAX_SNAPSHOT_STRING_CHARS)
      || ids.has(item.id)
      || !boundedString(item.group, MAX_SNAPSHOT_STRING_CHARS, true)
      || !boundedString(item.key, MAX_SNAPSHOT_STRING_CHARS)
      || !boundedString(item.rawValue, MAX_SNAPSHOT_STRING_CHARS, true)
      || (item.status !== "needs_decision" && item.status !== "different" && item.status !== "unsupported")
      || !(item.target === null || boundedString(item.target, MAX_SNAPSHOT_STRING_CHARS))
      || !boundedString(item.reason, MAX_SNAPSHOT_STRING_CHARS)) return null;
    const targetDescriptor = parseTargetDescriptor(item.targetDescriptor);
    if (targetDescriptor === undefined) return null;
    const base = { ...item, targetDescriptor, decision: null } as Cm1DecisionWorkbookItem;
    const decision = parseDecision(item.decision, base);
    if (decision === undefined) return null;
    ids.add(item.id);
    items.push({ ...base, decision });
  }
  if (source.mappingFingerprint !== fingerprint(items)) return null;
  if (source.descriptorFingerprint !== descriptorFingerprint(items)) return null;
  return { ...record, source: source as Cm1DecisionWorkbook["source"], items } as Cm1DecisionWorkbook;
}

export function serializeCm1DecisionWorkbook(workbook: Cm1DecisionWorkbook): string {
  const parsed = parseCm1DecisionWorkbook(workbook);
  if (!parsed) throw new Error("Refusing to export an invalid CM1 decision workbook.");
  const json = JSON.stringify(parsed, null, 2);
  if (new TextEncoder().encode(json).byteLength > CM1_DECISION_WORKBOOK_MAX_JSON_BYTES) {
    throw new Error("The CM1 decision workbook exceeds the portable JSON size limit.");
  }
  return json;
}

function sameSource(left: Cm1DecisionWorkbook, right: Cm1DecisionWorkbook): boolean {
  return left.source.datasetIdentity === right.source.datasetIdentity
    && left.source.mappingFingerprint === right.source.mappingFingerprint
    && left.source.descriptorFingerprint === right.source.descriptorFingerprint
    && mappingSnapshot(left.items) === mappingSnapshot(right.items)
    && descriptorWorkbookSnapshot(left.items) === descriptorWorkbookSnapshot(right.items);
}

export function importCm1DecisionWorkbook(
  json: string,
  expectedSource: Cm1DecisionWorkbook,
): Cm1DecisionWorkbook {
  if (new TextEncoder().encode(json).byteLength > CM1_DECISION_WORKBOOK_MAX_JSON_BYTES) {
    throw new Error("This workbook exceeds the portable JSON size limit.");
  }
  let value: unknown;
  try { value = JSON.parse(json); } catch { throw new Error("This file is not valid JSON."); }
  const parsed = parseCm1DecisionWorkbook(value);
  if (!parsed) throw new Error("This file is not a supported CM1 decision workbook.");
  if (!sameSource(parsed, expectedSource)) {
    throw new Error("This workbook belongs to a different CM1 source or mapping report.");
  }
  return parsed;
}

function storageKey(workbook: Cm1DecisionWorkbook): string {
  return `nimbus:cm1-decision-workbook:v2:${fnv1a(workbook.source.datasetIdentity)}:${workbook.source.mappingFingerprint}:${workbook.source.descriptorFingerprint}`;
}

export function loadCm1DecisionWorkbook(
  storage: WorkbookStorageReader,
  expectedSource: Cm1DecisionWorkbook,
): Cm1DecisionWorkbook | null {
  try {
    const json = storage.getItem(storageKey(expectedSource));
    return json ? importCm1DecisionWorkbook(json, expectedSource) : null;
  } catch {
    return null;
  }
}

export function saveCm1DecisionWorkbook(storage: WorkbookStorageWriter, workbook: Cm1DecisionWorkbook): void {
  storage.setItem(storageKey(workbook), serializeCm1DecisionWorkbook(workbook));
}
