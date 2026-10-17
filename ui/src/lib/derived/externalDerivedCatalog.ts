import type { StepInfo } from "@/lib/run/runOutputs";
import type {
  ExternalDerivedProvider,
  ExternalProviderRuntimeStatus,
} from "./externalDerivedProviders";

export interface ExternalProviderCatalogAssessment {
  provider: ExternalDerivedProvider;
  status: "available" | "disabled" | "runtime_unavailable" | "missing_inputs" | "unit_mismatch" | "collision";
  reason: string;
}

export function assessExternalProviderForStep(
  step: StepInfo,
  provider: ExternalDerivedProvider,
  runtime: ExternalProviderRuntimeStatus,
): ExternalProviderCatalogAssessment {
  if (!provider.enabled) return { provider, status: "disabled", reason: "Provider is disabled." };
  if (!runtime.available) return { provider, status: "runtime_unavailable", reason: runtime.reason ?? `${runtime.label} is unavailable.` };
  if (step.fields.includes(provider.output.fieldId)) {
    return { provider, status: "collision", reason: `Existing field '${provider.output.fieldId}' takes precedence.` };
  }
  const missing = provider.inputs.filter(({ fieldId }) => !step.fields.includes(fieldId));
  if (missing.length) return { provider, status: "missing_inputs", reason: `Missing ${missing.map(({ fieldId }) => fieldId).join(", ")}.` };
  const mismatched = provider.inputs.filter(({ fieldId, unit }) => unit != null && step.fieldUnits[fieldId] !== unit);
  if (mismatched.length) {
    return {
      provider,
      status: "unit_mismatch",
      reason: `Unit mismatch for ${mismatched.map(({ fieldId, unit }) => `${fieldId} (needs ${unit}, has ${step.fieldUnits[fieldId] ?? "unknown"})`).join(", ")}.`,
    };
  }
  return { provider, status: "available", reason: "Runtime, dependencies, units, and output identity qualify." };
}

export function augmentStepWithExternalProviders(
  step: StepInfo,
  providers: readonly ExternalDerivedProvider[],
  runtimeStatus: (provider: ExternalDerivedProvider) => ExternalProviderRuntimeStatus,
): StepInfo {
  const fields = [...step.fields];
  const fieldUnits = { ...step.fieldUnits };
  const fieldOrigins = { ...step.fieldOrigins };
  for (const provider of providers) {
    if (assessExternalProviderForStep({ ...step, fields, fieldUnits, fieldOrigins }, provider, runtimeStatus(provider)).status !== "available") continue;
    fields.push(provider.output.fieldId);
    fieldUnits[provider.output.fieldId] = provider.output.unit;
    fieldOrigins[provider.output.fieldId] = "external_provider";
  }
  return { ...step, fields, fieldUnits, fieldOrigins };
}

