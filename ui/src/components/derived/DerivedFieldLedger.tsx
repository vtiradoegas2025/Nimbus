import { useMemo } from "react";
import type { LocalDatasetSource } from "@/lib/dataset/datasetSources";
import { datasetSourceIdentity } from "@/lib/dataset/datasetSources";
import {
  DERIVED_FIELD_REGISTRY_SCHEMA,
  DERIVED_FIELD_REGISTRY_VERSION,
  assessDerivedFields,
  buildDerivedFieldProvenance,
  derivedFieldDefinition,
} from "@/lib/derived/derivedFieldRegistry";
import type { StepInfo } from "@/lib/run/runOutputs";

interface DerivedFieldLedgerProps {
  dataset: LocalDatasetSource;
  steps: readonly StepInfo[];
  stepIdx: number;
}

function nativeCatalog(step: StepInfo): { fields: Set<string>; units: Record<string, string> } {
  const fields = new Set(step.fields.filter((field) =>
    step.fieldOrigins ? step.fieldOrigins[field] !== "studio_derived" : derivedFieldDefinition(field) == null
  ));
  return { fields, units: Object.fromEntries([...fields].map((field) => [field, step.fieldUnits[field] ?? ""])) };
}

function safeName(value: string): string {
  return value.replace(/[^a-zA-Z0-9_-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 100) || "dataset";
}

function downloadJson(contents: string, filename: string): void {
  const url = URL.createObjectURL(new Blob([contents], { type: "application/json" }));
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = filename;
  document.body.appendChild(anchor); anchor.click(); anchor.remove(); window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function blocker(missing: readonly string[], incompatible: readonly { field: string; expected: string; actual: string | null }[]): string {
  const parts: string[] = [];
  if (missing.length) parts.push(`missing ${missing.join(" + ")}`);
  if (incompatible.length) parts.push(incompatible.map(({ field, expected, actual }) => `${field}: ${actual || "unit missing"}, needs ${expected}`).join("; "));
  return parts.join(" · ");
}

export function DerivedFieldLedger({ dataset, steps, stepIdx }: DerivedFieldLedgerProps) {
  const currentStep = steps[stepIdx] ?? steps[0];
  const assessments = useMemo(() => {
    if (!currentStep) return [];
    const native = nativeCatalog(currentStep);
    return assessDerivedFields(native.fields, native.units);
  }, [currentStep]);
  if (!currentStep) return null;
  const availableCount = assessments.filter(({ status }) => status === "available").length;
  const nativeCount = assessments.filter(({ status }) => status === "native_precedence").length;

  const exportReport = () => {
    const report = {
      schema: "nimbus.studio.derived-field-catalog-report",
      schemaVersion: 1,
      generatedAt: new Date().toISOString(),
      dataset,
      datasetIdentity: datasetSourceIdentity(dataset),
      registry: { schema: DERIVED_FIELD_REGISTRY_SCHEMA, version: DERIVED_FIELD_REGISTRY_VERSION },
      frames: steps.map((step) => {
        const native = nativeCatalog(step);
        return {
          frameIndex: step.index,
          simulationTimeS: step.simulationTimeS,
          publicationGeneration: step.publicationGeneration,
          fields: assessDerivedFields(native.fields, native.units).map((assessment) => ({
            status: assessment.status,
            missingDependencies: assessment.missingDependencies,
            incompatibleUnits: assessment.incompatibleUnits,
            ...(assessment.status === "available"
              ? { provenance: buildDerivedFieldProvenance(dataset, step, assessment.definition.id) }
              : { definition: assessment.definition }),
          })),
        };
      }),
    };
    downloadJson(`${JSON.stringify(report, null, 2)}\n`, `${safeName(dataset.displayName)}-derived-fields.json`);
  };

  return <details className="plot-studio__field-contract derived-ledger">
    <summary>
      <span className="plot-studio__field-contract-chevron" aria-hidden="true">›</span>
      <strong>Derived field registry</strong>
      <span>{availableCount} available now · {nativeCount} native override{nativeCount === 1 ? "" : "s"} · v{DERIVED_FIELD_REGISTRY_VERSION}</span>
    </summary>
    <div className="plot-studio__field-contract-body derived-ledger__body">
      <div className="derived-ledger__intro">
        <p>One dependency and provenance contract applies across supported data sources. Native output wins; Studio derives only when canonical fields and units qualify.</p>
        <button type="button" onClick={exportReport}>Export registry evidence</button>
      </div>
      <div className="derived-ledger__table" role="table" aria-label="Source-neutral derived field availability">
        {assessments.map(({ definition, status, missingDependencies, incompatibleUnits }) => (
          <div className="derived-ledger__row" role="row" key={definition.id}>
            <div role="cell"><strong>{definition.label}</strong><span>{definition.id} · definition v{definition.definitionVersion}</span></div>
            <span role="cell" className={`derived-ledger__status derived-ledger__status--${status}`}>{status === "available" ? "Studio-derived" : status === "native_precedence" ? "Native wins" : "Blocked"}</span>
            <div role="cell"><strong>{definition.dependencies.map(({ field, unit }) => `${field} [${unit}]`).join(" + ")}</strong><span>{status === "blocked" ? blocker(missingDependencies, incompatibleUnits) : definition.formula}</span></div>
            <div role="cell"><strong>{definition.unit}</strong><span>{definition.method}</span></div>
            <div role="cell" className="derived-ledger__parameters"><strong>{definition.parameters.length ? "Fixed assumptions" : "No parameters"}</strong><span>{definition.parameters.length ? definition.parameters.map(({ label, value, unit }) => `${label}: ${value}${unit ? ` ${unit}` : ""}`).join(" · ") : "Pure component magnitude"}</span></div>
          </div>
        ))}
      </div>
      <p className="derived-ledger__boundary">Built-in evaluators are qualified here. Dataset-local expressions run in cancellable workers. Permissioned, version-pinned Python providers can return validated fields through the Providers instrument; native plugin manifests remain disabled until a host adapter is connected.</p>
    </div>
  </details>;
}
