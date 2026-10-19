import { uniqueCm1NamelistValue } from "./cm1NamelistSummary";
import type { Cm1NamelistReport, Cm1SoundingReport } from "@/lib/dataset/datasetSources";

const GRAVITY_MS2 = 9.81;
const DRY_AIR_GAS_CONSTANT_JKGK = 287;
const SPECIFIC_HEAT_CP_JKGK = 1_004;
const REFERENCE_PRESSURE_PA = 100_000;
const WATER_VAPOR_MIXING_EPSILON = 0.622;
const MIN_VAPOR_PRESSURE_PA = 1e-6;

export interface Cm1ProjectedSoundingLevel {
  heightM: number;
  pressureHpa: number;
  temperatureK: number;
  dewpointK: number;
  windSpeedMs: number;
  windDirectionDeg: number;
}

export interface Cm1SoundingProjection {
  schema: "nimbus.studio.cm1-sounding-projection";
  schemaVersion: 1;
  sourcePath: string;
  targetFormat: "TMV_SHARPY_PROFILE_V1";
  copyEligible: boolean;
  exactCm1Equivalent: false;
  method: string;
  levels: Cm1ProjectedSoundingLevel[];
  text: string;
  blockers: string[];
  caveats: string[];
}

function parseScalar(raw: string | null): number | null {
  if (raw == null) return null;
  const normalized = raw.trim().replace(/[dD]/g, "e");
  if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(normalized)) return null;
  const value = Number(normalized);
  return Number.isFinite(value) ? value : null;
}

function vaporPressureFromMixingRatio(mixingRatioKgkg: number, pressurePa: number): number {
  return mixingRatioKgkg * pressurePa / (WATER_VAPOR_MIXING_EPSILON + mixingRatioKgkg);
}

function dewpointFromVaporPressure(vaporPressurePa: number): number {
  const logarithm = Math.log(Math.max(vaporPressurePa, MIN_VAPOR_PRESSURE_PA) / 611.21);
  return 273.15 + (243.04 * logarithm) / (17.625 - logarithm);
}

function windDirectionFromComponents(uMs: number, vMs: number): number {
  if (Math.hypot(uMs, vMs) <= 1e-12) return 0;
  const degrees = Math.atan2(-uMs, -vMs) * 180 / Math.PI;
  return (degrees + 360) % 360;
}

function fixed(value: number): string {
  const normalized = Math.abs(value) < 0.5e-6 ? 0 : value;
  return normalized.toFixed(6);
}

/**
 * Projects CM1's theta/qv/u/v profile into Nimbus's accepted text profile.
 * Pressure is reconstructed from the source surface pressure with a
 * trapezoidal virtual-potential-temperature integration of hydrostatic Exner
 * pressure. The result is an auditable interchange artifact, not a claim that
 * Nimbus and CM1 initialize or evolve an identical column.
 */
export function buildCm1SoundingProjection(
  report: Cm1SoundingReport,
  namelist?: Cm1NamelistReport,
): Cm1SoundingProjection {
  const blockers = [...report.warnings.map((warning) => `Source warning: ${warning}`)];
  let structurallyValid = report.warnings.length === 0;
  const caveats = [
    "Pressure is reconstructed hydrostatically from CM1 theta and qv; it is not copied from a CM1 model state.",
    "Nimbus interpolates the text profile to its grid and re-integrates pressure during initialization.",
    "Zero qv values use a finite near-dry dewpoint because the target text format requires one.",
  ];
  const { surface, levels } = report;

  if (!(surface.pressureHpa > 0)) {
    blockers.push("Surface pressure must be positive.");
    structurallyValid = false;
  }
  if (!(surface.potentialTemperatureK > 0)) {
    blockers.push("Surface potential temperature must be positive.");
    structurallyValid = false;
  }
  if (surface.mixingRatioGkg < 0) {
    blockers.push("Surface water-vapor mixing ratio must not be negative.");
    structurallyValid = false;
  }
  if (levels.length < 2) {
    blockers.push("At least two above-surface levels are required by the Nimbus text reader.");
    structurallyValid = false;
  }

  levels.forEach((level, index) => {
    if (level.heightM < 0) {
      blockers.push(`Level ${index + 1} has a negative height.`);
      structurallyValid = false;
    }
    if (index > 0 && level.heightM <= levels[index - 1].heightM) {
      blockers.push(`Level ${index + 1} does not increase strictly in height.`);
      structurallyValid = false;
    }
    if (!(level.potentialTemperatureK > 0)) {
      blockers.push(`Level ${index + 1} has non-positive potential temperature.`);
      structurallyValid = false;
    }
    if (level.mixingRatioGkg < 0) {
      blockers.push(`Level ${index + 1} has negative water-vapor mixing ratio.`);
      structurallyValid = false;
    }
  });

  if (structurallyValid) {
    const surfaceExner = Math.pow(
      surface.pressureHpa * 100 / REFERENCE_PRESSURE_PA,
      DRY_AIR_GAS_CONSTANT_JKGK / SPECIFIC_HEAT_CP_JKGK,
    );
    const surfaceTemperatureK = surface.potentialTemperatureK * surfaceExner;
    const surfaceDewpointK = dewpointFromVaporPressure(vaporPressureFromMixingRatio(
      surface.mixingRatioGkg / 1_000,
      surface.pressureHpa * 100,
    ));
    if (surfaceDewpointK > surfaceTemperatureK + 0.05) {
      blockers.push("The CM1 surface header projects above water saturation; Nimbus would clamp its moisture.");
    }
  }

  const isndMatches = namelist
    ? Object.keys(namelist.values).filter((identity) => identity.toLowerCase().endsWith(".isnd"))
    : [];
  const rawIsnd = namelist ? uniqueCm1NamelistValue(namelist, "isnd") : null;
  const isnd = parseScalar(rawIsnd);
  if (isndMatches.length > 1) {
    blockers.push("CM1 isnd is ambiguous across namelist groups; wind ownership cannot be reviewed safely.");
  } else if (rawIsnd != null && isnd == null) {
    blockers.push("CM1 isnd is not one finite numeric mode; external-sounding behavior cannot be reviewed safely.");
  }
  if (isnd === 17) {
    blockers.push("CM1 isnd=17 ignores input_sounding winds, but Nimbus file soundings currently consume their wind column.");
  } else if (isnd != null && isnd !== 7) {
    caveats.push(`CM1 isnd=${isnd} does not select the external profile as the active CM1 sounding.`);
  }

  const projected: Cm1ProjectedSoundingLevel[] = [];
  if (structurallyValid) {
    const kappa = DRY_AIR_GAS_CONSTANT_JKGK / SPECIFIC_HEAT_CP_JKGK;
    let previousHeightM = 0;
    let previousVirtualTheta = surface.potentialTemperatureK
      * (1 + 0.608 * surface.mixingRatioGkg / 1_000);
    let exner = Math.pow(surface.pressureHpa * 100 / REFERENCE_PRESSURE_PA, kappa);

    for (let index = 0; index < levels.length; index += 1) {
      const source = levels[index];
      const mixingRatioKgkg = source.mixingRatioGkg / 1_000;
      const virtualTheta = source.potentialTemperatureK * (1 + 0.608 * mixingRatioKgkg);
      const layerDepthM = source.heightM - previousHeightM;
      exner -= GRAVITY_MS2 / SPECIFIC_HEAT_CP_JKGK * layerDepthM
        * 0.5 * (1 / previousVirtualTheta + 1 / virtualTheta);
      if (!(exner > 0) || !Number.isFinite(exner)) {
        blockers.push(`Hydrostatic pressure became invalid at source line ${source.sourceLine}.`);
        break;
      }
      const pressurePa = REFERENCE_PRESSURE_PA * Math.pow(exner, 1 / kappa);
      const temperatureK = source.potentialTemperatureK * exner;
      const vaporPressurePa = vaporPressureFromMixingRatio(mixingRatioKgkg, pressurePa);
      const dewpointK = dewpointFromVaporPressure(vaporPressurePa);
      if (![pressurePa, temperatureK, dewpointK].every(Number.isFinite)) {
        blockers.push(`Thermodynamic projection became non-finite at source line ${source.sourceLine}.`);
        break;
      }
      if (dewpointK > temperatureK + 0.05) {
        blockers.push(`Source line ${source.sourceLine} projects above water saturation; Nimbus would clamp its moisture.`);
      }
      projected.push({
        heightM: source.heightM,
        pressureHpa: pressurePa / 100,
        temperatureK,
        dewpointK,
        windSpeedMs: Math.hypot(source.uMs, source.vMs),
        windDirectionDeg: windDirectionFromComponents(source.uMs, source.vMs),
      });
      previousHeightM = source.heightM;
      previousVirtualTheta = virtualTheta;
    }
  }

  const copyEligible = blockers.length === 0 && projected.length === levels.length;
  const lines = [
    "TMV_SHARPY_PROFILE_V1",
    "# Projected by Nimbus Studio from CM1 input_sounding",
    "# Review the source compatibility report; this is not an exact CM1 initialization.",
    "station_id\tCM1_IMPORT",
    `levels\t${projected.length}`,
    "h\tp\tt\ttd\twspd\twdir",
    ...projected.map((level) => [
      fixed(level.heightM),
      fixed(level.pressureHpa),
      fixed(level.temperatureK),
      fixed(level.dewpointK),
      fixed(level.windSpeedMs),
      fixed(level.windDirectionDeg),
    ].join("\t")),
  ];

  return {
    schema: "nimbus.studio.cm1-sounding-projection",
    schemaVersion: 1,
    sourcePath: report.path,
    targetFormat: "TMV_SHARPY_PROFILE_V1",
    copyEligible,
    exactCm1Equivalent: false,
    method: "Hydrostatic Exner integration using virtual potential temperature",
    levels: projected,
    text: `${lines.join("\n")}\n`,
    blockers,
    caveats,
  };
}
