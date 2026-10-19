import type { Cm1NamelistReport } from "@/lib/dataset/datasetSources";

export interface Cm1NamelistSummary {
  controls: string[];
  inspection: string;
}

export function uniqueCm1NamelistValue(
  report: Cm1NamelistReport,
  key: string,
): string | null {
  const suffix = `.${key.toLowerCase()}`;
  const matches = Object.entries(report.values).filter(([identity]) =>
    identity.toLowerCase().endsWith(suffix),
  );
  return matches.length === 1 ? matches[0][1].trim() : null;
}

function compactRawValue(value: string): string {
  const oneLine = value.replace(/\s+/g, " ").trim();
  return oneLine.length <= 72 ? oneLine : `${oneLine.slice(0, 69)}…`;
}

function scalarControl(
  report: Cm1NamelistReport,
  key: string,
  label: string,
): string | null {
  const value = uniqueCm1NamelistValue(report, key);
  return value == null ? null : `${label} ${compactRawValue(value)}`;
}

/**
 * Produces a conservative, read-only synopsis of a parsed CM1 namelist.
 * Values are omitted when the same key appears in more than one group, because
 * guessing which group owns a scientific control would be misleading.
 */
export function summarizeCm1Namelist(
  report: Cm1NamelistReport,
): Cm1NamelistSummary {
  const nx = uniqueCm1NamelistValue(report, "nx");
  const ny = uniqueCm1NamelistValue(report, "ny");
  const nz = uniqueCm1NamelistValue(report, "nz");
  const dx = uniqueCm1NamelistValue(report, "dx");
  const dy = uniqueCm1NamelistValue(report, "dy");
  const dz = uniqueCm1NamelistValue(report, "dz");
  const runTime = uniqueCm1NamelistValue(report, "run_time");
  const controls = [
    nx && ny && nz
      ? `grid ${compactRawValue(nx)} × ${compactRawValue(ny)} × ${compactRawValue(nz)}`
      : null,
    dx && dy && dz
      ? `spacing ${compactRawValue(dx)} / ${compactRawValue(dy)} / ${compactRawValue(dz)}`
      : null,
    runTime == null
      ? scalarControl(report, "timax", "run time")
      : `run time ${compactRawValue(runTime)}`,
    scalarControl(report, "dtl", "dtl"),
    scalarControl(report, "tapfrq", "output cadence"),
    scalarControl(report, "ptype", "microphysics ptype"),
    scalarControl(report, "sgsmodel", "SGS model"),
    scalarControl(report, "output_format", "output format"),
  ].filter((value): value is string => value != null);
  const groupLabel = `${report.groups.length} namelist ${report.groups.length === 1 ? "group" : "groups"}`;
  const assignmentLabel = `${report.assignmentCount} ${report.assignmentCount === 1 ? "assignment" : "assignments"}`;
  const warningLabel = report.warnings.length > 0
    ? ` · ${report.warnings.length} parser ${report.warnings.length === 1 ? "warning" : "warnings"}`
    : "";
  return {
    controls,
    inspection: `${groupLabel} · ${assignmentLabel}${warningLabel}`,
  };
}
