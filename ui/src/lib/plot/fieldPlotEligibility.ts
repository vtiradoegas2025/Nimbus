import {
  registeredFieldRank,
  type StepInfo,
} from "@/lib/run/runOutputs";
import {
  CROSS_SECTION_VECTOR_CAPABILITY,
  NATIVE_REDUCTION_CAPABILITY,
  NATIVE_VOLUME_REDUCTION_CAPABILITY,
  PLANE_SCALAR_CAPABILITY,
  PLAN_VECTOR_CAPABILITY,
  VOLUME_SCALAR_CAPABILITY,
} from "./plotRegistry";
import { availableVectorPairs } from "@/lib/science/vectorComponents";

export type PlotFieldRank = 1 | 2 | 3;
export type PlotFieldOrigin = NonNullable<StepInfo["fieldOrigins"]>[string];

function hasAllowedRank(
  runDir: string,
  step: StepInfo,
  field: string,
  ranks: ReadonlySet<number>,
): boolean {
  const rank = registeredFieldRank(
    runDir,
    step.index,
    field,
    step.publicationGeneration,
  );
  return rank != null && ranks.has(rank);
}

/** Fields the selected frame can pass to a renderer with an exact rank
 * contract. Rank comes from the dataset adapter, not a name heuristic. */
export function fieldsForPlotRanks(
  runDir: string,
  step: StepInfo | null | undefined,
  allowedRanks: readonly PlotFieldRank[],
  allowedOrigins?: ReadonlySet<PlotFieldOrigin>,
): string[] {
  if (!step) return [];
  const ranks = new Set<number>(allowedRanks);
  return step.fields.filter((field) => {
    if (allowedOrigins && !allowedOrigins.has(step.fieldOrigins?.[field] ?? "native")) {
      return false;
    }
    return hasAllowedRank(runDir, step, field, ranks);
  });
}

/** Union of fields that keep the requested rank/provider contract in every
 * frame where they are published. A field that changes rank mid-run is not a
 * partially valid animation; it is withheld from that plot. */
export function fieldsAcrossStepsForPlotRanks(
  runDir: string,
  steps: readonly StepInfo[],
  allowedRanks: readonly PlotFieldRank[],
  allowedOrigins?: ReadonlySet<PlotFieldOrigin>,
): string[] {
  const fields = new Set(steps.flatMap((step) => step.fields));
  const ranks = new Set<number>(allowedRanks);
  return [...fields].filter((field) => {
    const present = steps.filter((step) => step.fields.includes(field));
    return present.length > 0 && present.every((step) => {
      if (allowedOrigins && !allowedOrigins.has(step.fieldOrigins?.[field] ?? "native")) {
        return false;
      }
      return hasAllowedRank(runDir, step, field, ranks);
    });
  }).sort();
}

/** True only when all named components are present at the same exact rank. */
export function fieldsSharePlotRank(
  runDir: string,
  step: StepInfo | null | undefined,
  fields: readonly string[],
  allowedRanks: readonly PlotFieldRank[],
): boolean {
  if (!step || fields.length === 0 || fields.some((field) => !step.fields.includes(field))) {
    return false;
  }
  const ranks = fields.map((field) => registeredFieldRank(
    runDir,
    step.index,
    field,
    step.publicationGeneration,
  ));
  return ranks[0] != null
    && allowedRanks.includes(ranks[0] as PlotFieldRank)
    && ranks.every((rank) => rank === ranks[0]);
}

/** Candidate fields whose adapters report one identical plottable rank across
 * every supplied dataset/frame. Used before any cellwise comparison or
 * ensemble statistic is offered. */
export function fieldsWithSharedPlotRank(
  candidates: readonly string[],
  frames: readonly { runDir: string; step: StepInfo | null | undefined }[],
  allowedRanks: readonly PlotFieldRank[],
): string[] {
  if (frames.length === 0 || frames.some(({ step }) => !step)) return [];
  return candidates.filter((field) => {
    const ranks = frames.map(({ runDir, step }) => registeredFieldRank(
      runDir,
      step!.index,
      field,
      step!.publicationGeneration,
    ));
    return ranks[0] != null
      && allowedRanks.includes(ranks[0] as PlotFieldRank)
      && ranks.every((rank) => rank === ranks[0]);
  });
}

/** Add renderer capabilities to the ordinary field union used by the picker.
 * The sentinels never enter a field menu; they only make registry gating obey
 * the same rank/provider contract as the mounted renderer. */
export function addPlotFieldCapabilities(
  available: Set<string>,
  runDir: string,
  steps: readonly StepInfo[],
): Set<string> {
  let hasPlane = false;
  let hasVolume = false;
  let hasNative = false;
  let hasNativeVolume = false;
  let hasPlanVector = false;
  let hasCrossSectionVector = false;

  for (const step of steps) {
    for (const field of step.fields) {
      const rank = registeredFieldRank(
        runDir,
        step.index,
        field,
        step.publicationGeneration,
      );
      const native = (step.fieldOrigins?.[field] ?? "native") === "native";
      if (rank === 2 || rank === 3) hasPlane = true;
      if (rank === 3) hasVolume = true;
      // The native reducer bridge validates grid-shaped Nimbus output. A bare
      // rank-1 provider/profile is not a native history field even though the
      // arithmetic reducer itself could scan a flat buffer.
      if (native && (rank === 2 || rank === 3)) hasNative = true;
      if (native && rank === 3) hasNativeVolume = true;
    }
    hasPlanVector ||= availableVectorPairs(2, step.fields).some((pair) =>
      fieldsSharePlotRank(runDir, step, [pair.xField, pair.yField], [2, 3])
    );
    hasCrossSectionVector ||= availableVectorPairs(1, step.fields).some((pair) =>
      fieldsSharePlotRank(runDir, step, [pair.xField, pair.yField], [3])
    );
  }

  if (hasPlane) available.add(PLANE_SCALAR_CAPABILITY);
  if (hasVolume) available.add(VOLUME_SCALAR_CAPABILITY);
  if (hasNative) available.add(NATIVE_REDUCTION_CAPABILITY);
  if (hasNativeVolume) available.add(NATIVE_VOLUME_REDUCTION_CAPABILITY);
  if (hasPlanVector) available.add(PLAN_VECTOR_CAPABILITY);
  if (hasCrossSectionVector) available.add(CROSS_SECTION_VECTOR_CAPABILITY);
  return available;
}
