import { Icon } from "@/components/shell/Icon";
import {
  compressionDisclosures,
  compressionDisclosuresAcrossSteps,
} from "@/lib/plot/compressionProvenance";
import type { StepInfo } from "@/lib/run/runOutputs";
import { useOutputHierarchy } from "@/lib/run/useOutputHierarchy";
import { ControlGroup } from "./common";
import { useRef } from "react";

export function PlotCompressionControls({
  step,
  steps,
  fields,
}: {
  step?: StepInfo | null;
  steps?: readonly StepInfo[];
  fields: readonly (string | null | undefined)[];
}) {
  const disclosures = steps
    ? compressionDisclosuresAcrossSteps(steps, fields)
    : compressionDisclosures(step, fields);
  if (disclosures.length === 0) return null;
  const title = disclosures.map((entry) =>
    `${entry.fields.join(", ")}: ${entry.detail}${entry.stepIndices
      ? ` · steps ${entry.stepIndices.join(", ")}`
      : ""}`
  ).join("\n");
  const fieldCount = disclosures.reduce((count, entry) => count + entry.fields.length, 0);
  const label = disclosures.length === 1
    ? disclosures[0]!.shortLabel
    : `${fieldCount} lossy fields · ${disclosures.length} contracts`;
  return (
    <ControlGroup label="Storage fidelity">
      <span className="plot-source-disclosure" title={title}>
        <Icon name="info" size={11} />
        <span>{label}</span>
      </span>
    </ControlGroup>
  );
}

/** Temporal/aggregate products deliberately stay on the root history until a
 * moving child has an explicit cross-frame Earth-coordinate analysis policy. */
export function RootHistoryScopeNotice({
  runDir,
  step,
  label = "Root history across time",
}: {
  runDir: string;
  step: StepInfo | null | undefined;
  label?: string;
}) {
  const eligible = step?.publicationState === "complete"
    && step.publicationGeneration != null;
  const hierarchy = useOutputHierarchy(runDir, step ?? null, eligible);
  const seen = useRef({ runDir, nested: false });
  if (seen.current.runDir !== runDir) seen.current = { runDir, nested: false };
  if (hierarchy.levels.length > 0) seen.current.nested = true;
  if (!seen.current.nested) return null;
  return (
    <div
      className="plot-scope-notice"
      title="Nested children can move and publish on a different cadence. This product remains on the co-temporal root history instead of silently mixing grid footprints."
    >
      <Icon name="info" size={11} />
      <span>{label}</span>
    </div>
  );
}
