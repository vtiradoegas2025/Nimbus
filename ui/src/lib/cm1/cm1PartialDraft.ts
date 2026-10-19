import type { ConfigObject, ConfigValue, FieldSchema } from "@/lib/config/configSchema";
import { qualifyKey, SCHEMA_SECTIONS, writePath } from "@/lib/config/configSchema";
import type {
  Cm1CompatibilityEntry,
  Cm1CompatibilityReport,
  Cm1CompatibilityStatus,
} from "./cm1Compatibility";
import { stringifyYaml } from "@/lib/config/yaml";

export interface Cm1DraftValueCheck {
  entryId: string;
  target: string;
  accepted: boolean;
  message: string;
}

export interface Cm1DraftOmission {
  entryId: string;
  status: Cm1CompatibilityStatus;
  reason: string;
}

export interface Cm1PartialDraft {
  schema: "nimbus.studio.cm1-partial-draft";
  schemaVersion: 1;
  runnable: false;
  config: ConfigObject;
  yaml: string;
  includedEntryIds: string[];
  omissions: Cm1DraftOmission[];
  valueChecks: Cm1DraftValueCheck[];
  blockers: string[];
}

const fieldsByPath = new Map<string, FieldSchema>();
const targetOrder = new Map<string, number>();
for (const section of SCHEMA_SECTIONS) {
  for (const field of section.fields) {
    const path = qualifyKey(section, field);
    fieldsByPath.set(path, field);
    targetOrder.set(path, targetOrder.size);
  }
}

function numeric(raw: string): number | null {
  const normalized = raw.trim().replace(/[dD]/g, "e");
  if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(normalized)) return null;
  const value = Number(normalized);
  return Number.isFinite(value) ? value : null;
}

function translatedValue(entry: Cm1CompatibilityEntry): ConfigValue | null {
  if (entry.key === "ptype" && numeric(entry.rawValue) === 0) return "none";
  return numeric(entry.rawValue);
}

function validateValue(
  entry: Cm1CompatibilityEntry,
  target: string,
  value: ConfigValue,
): Cm1DraftValueCheck {
  const field = fieldsByPath.get(target);
  if (!field) {
    return {
      entryId: entry.id,
      target,
      accepted: false,
      message: "The candidate target is not present in the current Studio configuration descriptor.",
    };
  }
  if (field.kind === "int" || field.kind === "float") {
    if (typeof value !== "number") {
      return { entryId: entry.id, target, accepted: false, message: "Nimbus requires a numeric scalar." };
    }
    if (field.kind === "int" && !Number.isInteger(value)) {
      return { entryId: entry.id, target, accepted: false, message: "Nimbus requires an integer value." };
    }
    if (field.min != null && value < field.min) {
      return { entryId: entry.id, target, accepted: false, message: `Value ${value} is below Studio's current minimum ${field.min}.` };
    }
    if (field.max != null && value > field.max) {
      return { entryId: entry.id, target, accepted: false, message: `Value ${value} exceeds Studio's current maximum ${field.max}.` };
    }
    return { entryId: entry.id, target, accepted: true, message: "Accepted by the current Studio field descriptor." };
  }
  if (field.kind === "enum") {
    if (typeof value !== "string") {
      return { entryId: entry.id, target, accepted: false, message: "Nimbus requires an enum token." };
    }
    const option = field.options.find((candidate) => candidate.value === value);
    if (!option) {
      return { entryId: entry.id, target, accepted: false, message: `Enum token '${value}' is not offered by the current simulator descriptor.` };
    }
    if (option.disabled) {
      return { entryId: entry.id, target, accepted: false, message: option.note ?? `Enum token '${value}' is disabled.` };
    }
    return { entryId: entry.id, target, accepted: true, message: "Accepted by the current Studio enum descriptor." };
  }
  return { entryId: entry.id, target, accepted: false, message: "This target type is not supported by the mechanical draft builder." };
}

/**
 * Builds a deterministic, intentionally incomplete YAML preview. Only Ready
 * mappings that pass the current Studio descriptor enter the config object.
 * It is never runnable and never fills unresolved sections with defaults.
 */
export function buildCm1PartialDraft(report: Cm1CompatibilityReport): Cm1PartialDraft {
  const candidates = report.entries
    .filter((entry) => entry.status === "ready" && entry.target != null)
    .sort((left, right) =>
      (targetOrder.get(left.target!) ?? Number.MAX_SAFE_INTEGER)
      - (targetOrder.get(right.target!) ?? Number.MAX_SAFE_INTEGER)
      || left.id.localeCompare(right.id),
    );
  let config: ConfigObject = { schema: "nimbus.v1" };
  const includedEntryIds: string[] = [];
  const valueChecks: Cm1DraftValueCheck[] = [];
  const failedReady = new Map<string, string>();
  for (const entry of candidates) {
    const target = entry.target!;
    const value = translatedValue(entry);
    const check = value == null
      ? { entryId: entry.id, target, accepted: false, message: "The source value is not one finite scalar." }
      : validateValue(entry, target, value);
    valueChecks.push(check);
    if (!check.accepted || value == null) {
      failedReady.set(entry.id, check.message);
      continue;
    }
    config = writePath(config, target, value);
    includedEntryIds.push(entry.id);
  }
  const omissions = report.entries
    .filter((entry) => !includedEntryIds.includes(entry.id))
    .map((entry): Cm1DraftOmission => ({
      entryId: entry.id,
      status: failedReady.has(entry.id) ? "needs_decision" : entry.status,
      reason: failedReady.get(entry.id) ?? entry.reason,
    }));
  const preamble = [
    "# Nimbus Studio CM1 mechanical draft preview",
    "# NON-RUNNABLE: contains only qualified Ready scalars; unresolved values are omitted.",
    "# Review the paired compatibility report before creating any runnable configuration.",
  ].join("\n");
  return {
    schema: "nimbus.studio.cm1-partial-draft",
    schemaVersion: 1,
    runnable: false,
    config,
    yaml: `${preamble}\n${stringifyYaml(config)}`,
    includedEntryIds,
    omissions,
    valueChecks,
    blockers: [
      ...report.draftBlockers,
      ...(failedReady.size > 0
        ? [`${failedReady.size} Ready candidate${failedReady.size === 1 ? " fails" : "s fail"} current Studio value checks`]
        : []),
      `${omissions.length} source assignment${omissions.length === 1 ? " is" : "s are"} omitted from the partial draft`,
    ],
  };
}
