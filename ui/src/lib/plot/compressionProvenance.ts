import type { FieldCompressionInfo, StepInfo } from "@/lib/run/runOutputs";

export interface CompressionDisclosure {
  fields: string[];
  shortLabel: string;
  detail: string;
  stepIndices?: number[];
}

/** All distinct contracts consumed by a temporal product. A run may legally
 * switch storage format or ZFP parameters between frames, so inspecting one
 * representative step would understate the plotted history. */
export function compressionDisclosuresAcrossSteps(
  steps: readonly StepInfo[],
  fields: readonly (string | null | undefined)[],
): CompressionDisclosure[] {
  const grouped = new Map<string, CompressionDisclosure>();
  for (const step of steps) {
    for (const disclosure of compressionDisclosures(step, fields)) {
      const key = `${disclosure.fields.join("\u0000")}\u0001${disclosure.detail}`;
      const existing = grouped.get(key);
      if (existing) existing.stepIndices?.push(step.index);
      else grouped.set(key, { ...disclosure, stepIndices: [step.index] });
    }
  }
  return [...grouped.values()];
}

function finiteLabel(value: number): string {
  if (!Number.isFinite(value)) return "undeclared";
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude < 1e-3 || magnitude >= 1e4)) {
    return value.toExponential(2).replace("e+", "e");
  }
  return Number(value.toPrecision(4)).toString();
}

function contractKey(info: FieldCompressionInfo, execution?: StepInfo["compressionExecution"]): string {
  return [
    info.codec,
    info.mode,
    info.modeParameter,
    info.modeParameterUnits,
    info.errorContract,
    info.sparseThreshold,
    info.float16Prefilter,
    info.decodedDtype,
    execution?.fieldExecutionPolicy ?? "",
    execution?.fieldWorkers ?? "",
    execution?.fieldScratchBudgetMib ?? "",
  ].join("\u0000");
}

function codecLabel(codec: string): string {
  return codec.toLowerCase() === "zfp" ? "ZFP" : codec.toUpperCase();
}

function describeContract(
  info: FieldCompressionInfo,
  execution?: StepInfo["compressionExecution"],
): Omit<CompressionDisclosure, "fields"> {
  const codec = codecLabel(info.codec);
  const parameter = `${finiteLabel(info.modeParameter)}${info.modeParameterUnits
    ? ` ${info.modeParameterUnits}`
    : ""}`;
  const shortLabel = `${codec} ${info.mode} · ${parameter}`;
  const additions = [
    info.errorContract ? `error contract: ${info.errorContract}` : null,
    info.sparseThreshold !== 0
      ? `sparse threshold: ${finiteLabel(info.sparseThreshold)}`
      : "sparse threshold off",
    info.float16Prefilter ? "float16 prefilter on" : "float16 prefilter off",
    info.decodedDtype ? `decoded ${info.decodedDtype}` : null,
    execution
      ? `field execution: ${execution.fieldExecutionPolicy} (${execution.fieldWorkers} worker${execution.fieldWorkers === 1 ? "" : "s"}, ${execution.fieldScratchBudgetMib} MiB scratch budget)`
      : null,
  ].filter((part): part is string => part != null);
  return { shortLabel, detail: `${shortLabel} · ${additions.join(" · ")}` };
}

/** Exact manifest-declared lossy contracts for the fields actually used by a plot.
 * Missing metadata means lossless/legacy and remains silent rather than guessed. */
export function compressionDisclosures(
  step: StepInfo | null | undefined,
  fields: readonly (string | null | undefined)[],
): CompressionDisclosure[] {
  const compression = step?.fieldCompression;
  if (!compression) return [];
  const grouped = new Map<string, { info: FieldCompressionInfo; fields: string[] }>();
  for (const field of [...new Set(fields.filter((name): name is string => Boolean(name)))].sort()) {
    const info = compression[field];
    if (!info) continue;
    const key = contractKey(info, step?.compressionExecution);
    const existing = grouped.get(key);
    if (existing) existing.fields.push(field);
    else grouped.set(key, { info, fields: [field] });
  }
  return [...grouped.values()].map(({ info, fields: contractFields }) => ({
    fields: contractFields,
    ...describeContract(info, step?.compressionExecution),
  }));
}
