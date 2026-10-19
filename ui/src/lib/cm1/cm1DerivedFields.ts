/** Compatibility facade for callers that predate the source-neutral derived
 * field registry. New code should import derivedFieldRegistry directly. */
import type { NpyN } from "@/lib/platform/npy";
import {
  DERIVED_FIELD_DEFINITIONS,
  assessDerivedFields,
  derivedFieldDefinition,
  evaluateDerivedField,
  type BuiltinDerivedFieldId,
  type DerivedFieldDefinition,
} from "@/lib/derived/derivedFieldRegistry";

export type Cm1DerivedFieldSpec = DerivedFieldDefinition;
export const CM1_DERIVED_FIELD_SPECS = DERIVED_FIELD_DEFINITIONS;

export function cm1DerivedFieldSpec(fieldName: string): DerivedFieldDefinition | null {
  return derivedFieldDefinition(fieldName);
}

export function availableCm1DerivedFields(fields: ReadonlySet<string>): DerivedFieldDefinition[] {
  const canonicalUnits: Record<string, string> = {
    theta: "K", p: "Pa", qv: "kg/kg", u: "m/s", v: "m/s",
  };
  return assessDerivedFields(fields, canonicalUnits)
    .filter(({ status }) => status === "available")
    .map(({ definition }) => definition);
}

export function deriveCm1Field(
  fieldName: BuiltinDerivedFieldId,
  inputs: Readonly<Record<string, NpyN>>,
): NpyN {
  return evaluateDerivedField(fieldName, inputs);
}
