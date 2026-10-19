import type { StepInfo } from "../run/runOutputs";

export interface HistoryReductionCache<T> {
  seriesKey: string;
  values: Map<string, T | null>;
  pending: Map<string, Promise<T | null>>;
  activeKeys: Set<string>;
}

export function newHistoryReductionCache<T>(seriesKey = ""): HistoryReductionCache<T> {
  return { seriesKey, values: new Map(), pending: new Map(), activeKeys: new Set() };
}

export function historyFrameKey(step: Pick<StepInfo, "index" | "publicationGeneration" | "fields">, fieldName: string): string {
  return JSON.stringify([step.publicationGeneration, step.index, step.fields.includes(fieldName)]);
}

export function reconcileHistoryReductionCache<T>(cache: HistoryReductionCache<T>, keys: Iterable<string>): void {
  cache.activeKeys = new Set(keys);
  for (const key of cache.values.keys()) if (!cache.activeKeys.has(key)) cache.values.delete(key);
  for (const key of cache.pending.keys()) if (!cache.activeKeys.has(key)) cache.pending.delete(key);
}

export function historySourcesAreAppendOnly(previous: Iterable<string>, current: ReadonlySet<string>): boolean {
  for (const key of previous) if (!current.has(key)) return false;
  return true;
}

export function historyUnitMismatch(steps: readonly Pick<StepInfo, "fields" | "fieldUnits">[], fieldName: string): string | null {
  const units = new Set(steps.filter(step => step.fields.includes(fieldName))
    .map(step => canonicalHistoryUnit(step.fieldUnits?.[fieldName])));
  if (units.size <= 1) return null;
  return `${fieldName} has different units across these frames (${[...units].map(unit => unit ?? "unknown").join(", ")}). Normalize the source units before combining them.`;
}

export function canonicalHistoryUnit(unit: string | null | undefined): string | null {
  const value = unit?.trim();
  if (!value) return null;
  const spelling = value.replace(/⁻¹/g, "^-1").replace(/\*\*/g, "^")
    .replace(/[·⋅\s]/g, "").toLowerCase();
  if (["m/s", "ms-1", "ms^-1", "meterpersecond", "metrepersecond"].includes(spelling)) return "m/s";
  if (["kg/kg", "kgkg-1", "kgkg^-1"].includes(spelling)) return "kg/kg";
  if (["1/s", "s-1", "s^-1"].includes(spelling)) return "1/s";
  if (["kelvin", "degk", "°k"].includes(spelling) || value === "K") return "K";
  if (["pascal", "pascals"].includes(spelling) || value === "Pa") return "Pa";
  if (["hectopascal", "hectopascals", "millibar", "millibars", "mb"].includes(spelling) || value === "hPa") return "hPa";
  if (["1", "dimensionless", "unitless"].includes(spelling)) return "1";
  return value;
}
