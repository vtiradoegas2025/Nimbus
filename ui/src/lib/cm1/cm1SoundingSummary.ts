import type { Cm1NamelistReport, Cm1SoundingReport } from "@/lib/dataset/datasetSources";
import { uniqueCm1NamelistValue } from "./cm1NamelistSummary";

export interface Cm1SoundingSummary {
  facts: string[];
  disposition: string;
  attention: string | null;
}

function rawNumber(value: string | null): number | null {
  if (value == null) return null;
  const normalized = value.trim().replace(/[dD]/g, "e");
  if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(normalized)) {
    return null;
  }
  const parsed = Number(normalized);
  return Number.isFinite(parsed) ? parsed : null;
}

function rawLogical(value: string | null): boolean | null {
  if (value == null) return null;
  const normalized = value.trim().toLowerCase();
  if (normalized === ".true." || normalized === "true" || normalized === "t") return true;
  if (normalized === ".false." || normalized === "false" || normalized === "f") return false;
  return null;
}

function modelTopM(report: Cm1NamelistReport | undefined): number | null {
  if (!report) return null;
  const stretched = rawLogical(uniqueCm1NamelistValue(report, "stretch_z"));
  const ztop = rawNumber(uniqueCm1NamelistValue(report, "ztop"));
  const nz = rawNumber(uniqueCm1NamelistValue(report, "nz"));
  const dz = rawNumber(uniqueCm1NamelistValue(report, "dz"));
  const uniformTop = nz != null && dz != null ? nz * dz : null;
  if (stretched === true) return ztop;
  if (stretched === false) return uniformTop;
  if (ztop == null) return uniformTop;
  if (uniformTop == null) return ztop;
  return Math.abs(ztop - uniformTop) <= Math.max(1e-6, Math.abs(ztop) * 1e-9)
    ? ztop
    : null;
}

function compact(value: number, maximumFractionDigits = 2): string {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits }).format(value);
}

/** Summarizes the CM1-native profile without converting it into Nimbus input. */
export function summarizeCm1Sounding(
  report: Cm1SoundingReport,
  namelist?: Cm1NamelistReport,
): Cm1SoundingSummary {
  const peakWind = report.levels.reduce(
    (peak, level) => Math.max(peak, Math.hypot(level.uMs, level.vMs)),
    0,
  );
  const facts = [
    `${report.levels.length} ${report.levels.length === 1 ? "level" : "levels"}`,
    `surface ${compact(report.surface.pressureHpa)} hPa`,
    `θ ${compact(report.surface.potentialTemperatureK)} K`,
    `qv ${compact(report.surface.mixingRatioGkg, 3)} g/kg`,
    report.topHeightM == null ? null : `top ${compact(report.topHeightM / 1_000, 3)} km`,
    report.levels.length === 0 ? null : `peak wind ${compact(peakWind)} m/s`,
  ].filter((value): value is string => value != null);

  const isnd = rawNumber(namelist ? uniqueCm1NamelistValue(namelist, "isnd") : null);
  const configuredUse = isnd === 7
    ? "CM1 isnd=7 uses thermodynamics and winds from this file"
    : isnd === 17
      ? "CM1 isnd=17 uses thermodynamics but ignores this file’s winds"
      : isnd == null
        ? null
        : `CM1 isnd=${compact(isnd)} does not select the external-sounding modes 7 or 17`;
  const domainTop = modelTopM(namelist);
  const coversTop = report.topHeightM != null && domainTop != null
    ? report.topHeightM > domainTop
    : null;
  const coverage = coversTop === true
    ? `profile extends above the ${compact(domainTop! / 1_000, 3)} km model top`
    : coversTop === false
      ? `profile top must exceed the ${compact(domainTop! / 1_000, 3)} km model top`
      : null;
  const attention = [
    configuredUse && isnd !== 7 && isnd !== 17 ? configuredUse : null,
    coversTop === false ? coverage : null,
  ].filter((value): value is string => value != null).join(" · ") || null;
  const disposition = [
    configuredUse && (isnd === 7 || isnd === 17) ? configuredUse : null,
    coversTop === true ? coverage : null,
    "native file requires the reviewed Nimbus projection or a direct adapter before a run",
  ].filter((value): value is string => value != null).join(" · ");
  return { facts, disposition, attention };
}
