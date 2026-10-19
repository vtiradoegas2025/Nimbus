/* Field unit labels for plot colorbars. Units come from StepInfo.fieldUnits (field_contract.cpp). */

/** field name -> physical unit, as carried on StepInfo.fieldUnits. */
export type FieldUnits = Record<string, string> | null | undefined;

/** Compatibility for canonical fields emitted by older run manifests without
 * units. Keep this deliberately small: a guessed unit is worse than no unit. */
const LEGACY_CANONICAL_UNITS: Readonly<Record<string, string>> = {
  pbl_height: "m",
};

/** Physical unit for `field`, or null if unknown or dimensionless ("1" is suppressed). */
export function unitFor(field: string, units: FieldUnits): string | null {
  const u = units?.[field] ?? LEGACY_CANONICAL_UNITS[field];
  if (!u || u === "1") return null;
  return u;
}

/** `field (unit)` when a unit is known, otherwise the bare field name. */
export function labelWithUnit(field: string, units: FieldUnits): string {
  const u = unitFor(field, units);
  return u ? `${field} (${u})` : field;
}
