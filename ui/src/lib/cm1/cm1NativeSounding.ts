import type { Cm1SoundingReport } from "@/lib/dataset/datasetSources";

export type NativeCm1Coverage = "covers" | "shallow" | "unknown";

export interface NativeCm1SoundingSummary {
  levelCount: number;
  bottomHeightM: number | null;
  topHeightM: number | null;
  peakWindMs: number;
  coverage: NativeCm1Coverage;
  coverageMessage: string;
  hasEmbeddedWinds: boolean;
}

/**
 * Describe only facts present in a native CM1 input_sounding. This is a UI
 * inspection summary, not a conversion and not a claim of CM1/Nimbus state
 * equivalence.
 */
export function summarizeNativeCm1Sounding(
  report: Cm1SoundingReport,
  modelTopM: number | null,
): NativeCm1SoundingSummary {
  const finiteLevels = report.levels.filter((level) =>
    Number.isFinite(level.heightM)
    && Number.isFinite(level.uMs)
    && Number.isFinite(level.vMs),
  );
  const bottomHeightM = finiteLevels[0]?.heightM ?? null;
  const topHeightM = Number.isFinite(report.topHeightM)
    ? report.topHeightM
    : finiteLevels.at(-1)?.heightM ?? null;
  const peakWindMs = finiteLevels.reduce(
    (peak, level) => Math.max(peak, Math.hypot(level.uMs, level.vMs)),
    0,
  );
  const hasEmbeddedWinds = finiteLevels.length > 0;
  const coverage = modelTopM == null || !Number.isFinite(modelTopM) || topHeightM == null
    ? "unknown"
    : topHeightM >= modelTopM
      ? "covers"
      : "shallow";
  const coverageMessage = coverage === "covers"
    ? `Profile reaches ${formatHeight(topHeightM)} and covers the configured ${formatHeight(modelTopM)} model top.`
    : coverage === "shallow"
      ? `Profile stops at ${formatHeight(topHeightM)} below the configured ${formatHeight(modelTopM)} model top.`
      : "Model-top coverage will be checked by the exact simulator before launch.";
  return {
    levelCount: report.levels.length,
    bottomHeightM,
    topHeightM,
    peakWindMs,
    coverage,
    coverageMessage,
    hasEmbeddedWinds,
  };
}

export function formatHeight(heightM: number | null): string {
  if (heightM == null || !Number.isFinite(heightM)) return "unknown height";
  if (Math.abs(heightM) >= 1_000) {
    return `${(heightM / 1_000).toLocaleString(undefined, { maximumFractionDigits: 2 })} km`;
  }
  return `${heightM.toLocaleString(undefined, { maximumFractionDigits: 0 })} m`;
}
