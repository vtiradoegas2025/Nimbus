import { alignDatasets, type AlignmentStatus, type DatasetAlignmentReport } from "@/lib/dataset/datasetAlignment";
import { datasetSourceIdentity, type LocalDatasetSource } from "@/lib/dataset/datasetSources";
import type { StepInfo } from "@/lib/run/runOutputs";

export const ENSEMBLE_ALIGNMENT_SCHEMA = "nimbus.studio.ensemble-alignment" as const;
export const ENSEMBLE_ALIGNMENT_VERSION = 1 as const;
export const MAX_ENSEMBLE_MEMBERS = 12;

export interface EnsembleMemberInput {
  source: LocalDatasetSource;
  steps: StepInfo[];
}

export interface EnsembleTimeMatch {
  referenceFrameIndex: number;
  memberFrameIndices: number[];
  simulationTimeS: number;
}

export interface EnsembleMemberAlignment {
  identity: string;
  label: string;
  report: DatasetAlignmentReport | null;
}

export interface EnsembleAlignmentReport {
  schema: typeof ENSEMBLE_ALIGNMENT_SCHEMA;
  schemaVersion: typeof ENSEMBLE_ALIGNMENT_VERSION;
  status: AlignmentStatus;
  aggregateAllowed: boolean;
  referenceIdentity: string | null;
  memberIdentities: string[];
  eligibleFields: string[];
  gridShape: [number, number, number] | null;
  timeMatches: EnsembleTimeMatch[];
  members: EnsembleMemberAlignment[];
  blockers: string[];
  warnings: string[];
}

export function alignEnsembleMembers(inputs: readonly EnsembleMemberInput[]): EnsembleAlignmentReport {
  const reference = inputs[0];
  const identities = inputs.map(({ source }) => datasetSourceIdentity(source));
  const members: EnsembleMemberAlignment[] = inputs.map(({ source }, index) => ({
    identity: identities[index],
    label: source.displayName,
    report: index === 0 || !reference ? null : alignDatasets(reference.source, reference.steps, source, inputs[index].steps),
  }));
  const blockers: string[] = [];
  const warnings: string[] = [];
  if (inputs.length < 2) blockers.push("An ensemble needs a reference plus at least one additional member.");
  if (inputs.length > MAX_ENSEMBLE_MEMBERS) blockers.push(`Interactive ensemble analysis is bounded to ${MAX_ENSEMBLE_MEMBERS} members.`);
  if (new Set(identities).size !== identities.length) blockers.push("Every ensemble member must identify a distinct dataset.");
  if (!reference || reference.steps.length === 0) blockers.push("The reference member has no complete frames.");

  const pairReports = members.slice(1).flatMap((member) => member.report ? [member.report] : []);
  pairReports.forEach((report, index) => {
    const label = members[index + 1]?.label ?? `Member ${index + 2}`;
    if (!report.differenceAllowed) blockers.push(`${label}: ${report.blockers.join(" · ")}`);
    report.warnings.forEach((warning) => warnings.push(`${label}: ${warning}`));
  });

  let eligibleFields = reference ? [...new Set(reference.steps.flatMap((step) => step.fields))].sort() : [];
  for (const report of pairReports) {
    const allowed = new Set(report.differenceEligibleFields);
    eligibleFields = eligibleFields.filter((field) => allowed.has(field));
  }
  if (inputs.length >= 2 && eligibleFields.length === 0) blockers.push("No canonical field has stable matching units across every member.");

  const timeMatches: EnsembleTimeMatch[] = [];
  if (reference && pairReports.length === inputs.length - 1 && pairReports.every((report) => report.differenceAllowed)) {
    for (let referenceFrameIndex = 0; referenceFrameIndex < reference.steps.length; referenceFrameIndex += 1) {
      const pairMatches = pairReports.map((report) => report.time.matches.find((match) => match.leftFrameIndex === referenceFrameIndex));
      if (pairMatches.every((match) => match != null)) {
        timeMatches.push({
          referenceFrameIndex,
          memberFrameIndices: [referenceFrameIndex, ...pairMatches.map((match) => match!.rightFrameIndex)],
          simulationTimeS: reference.steps[referenceFrameIndex].simulationTimeS!,
        });
      }
    }
  }
  if (inputs.length >= 2 && timeMatches.length === 0) blockers.push("No physical time is unambiguously available across every member.");

  const gridShape = pairReports.length > 0 && pairReports.every((report) => report.grid.status === "exact")
    ? pairReports[0].grid.shape
    : null;
  const aggregateAllowed = blockers.length === 0;
  const exact = aggregateAllowed
    && pairReports.every((report) => report.verdict === "exact")
    && timeMatches.length === reference?.steps.length;
  return {
    schema: ENSEMBLE_ALIGNMENT_SCHEMA,
    schemaVersion: ENSEMBLE_ALIGNMENT_VERSION,
    status: aggregateAllowed ? (exact ? "exact" : "partial") : "blocked",
    aggregateAllowed,
    referenceIdentity: reference ? identities[0] : null,
    memberIdentities: identities,
    eligibleFields,
    gridShape,
    timeMatches,
    members,
    blockers: [...new Set(blockers)],
    warnings: [...new Set(warnings)],
  };
}
