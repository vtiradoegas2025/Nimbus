/*
 * Thermodynamic helpers for the Skew-T plot. All inputs/outputs in SI base units
 * (Pa, K, kg/kg dry-air mixing ratio) unless noted; display conversion at call site.
 */

export const R_D = 287.05; // J/(kg·K), dry air gas constant
export const R_V = 461.5; // J/(kg·K), water vapor gas constant
export const EPSILON = R_D / R_V; // ≈ 0.6219
export const C_P = 1004.5; // J/(kg·K), specific heat dry air @ const p
export const KAPPA = R_D / C_P; // ≈ 0.2854 (Exner exponent)
export const P_REF_PA = 100000; // 1000 hPa
export const T_ZERO = 273.15; // K, 0 °C
export const L_V = 2.501e6; // J/kg, latent heat of vaporization (~0 °C)

/* ---------- Unit conversion ---------- */

export type PressureUnit = "Pa" | "hPa";

/** Resolve a manifest pressure-unit label to "Pa" or "hPa", or null when absent/unknown. */
export function pressureUnitFromLabel(label: string | null | undefined): PressureUnit | null {
  if (!label) return null;
  const normalized = label.trim().toLowerCase();
  if (normalized === "pa" || normalized === "pascal" || normalized === "pascals") {
    return "Pa";
  }
  if (
    normalized === "hpa" ||
    normalized === "hectopascal" ||
    normalized === "hectopascals" ||
    normalized === "mb" ||
    normalized === "mbar"
  ) {
    return "hPa";
  }
  return null;
}

/** Detect Pa vs hPa from array values; threshold 5000 splits cleanly (sea-level ~105 kPa). */
export function detectPressureUnit(samples: ArrayLike<number>): PressureUnit {
  let maxVal = -Infinity;
  for (let i = 0; i < samples.length; i++) {
    const v = samples[i];
    if (Number.isFinite(v) && v > maxVal) maxVal = v;
  }
  if (!Number.isFinite(maxVal)) return "Pa";
  return maxVal > 5000 ? "Pa" : "hPa";
}

export function toPa(value: number, unit: PressureUnit): number {
  return unit === "Pa" ? value : value * 100;
}

export function toHPa(value: number, unit: PressureUnit): number {
  return unit === "Pa" ? value / 100 : value;
}

export function kelvinToCelsius(t: number): number {
  return t - T_ZERO;
}

export function celsiusToKelvin(t: number): number {
  return t + T_ZERO;
}

/* ---------- Core relations ---------- */

/** T = θ · (p/p_ref)^κ. theta K, p Pa → T K. */
export function tempFromTheta(thetaK: number, pPa: number): number {
  if (pPa <= 0) return NaN;
  return thetaK * Math.pow(pPa / P_REF_PA, KAPPA);
}

/** θ = T · (p_ref/p)^κ. T K, p Pa → theta K. */
export function thetaFromTemp(tK: number, pPa: number): number {
  if (pPa <= 0) return NaN;
  return tK * Math.pow(P_REF_PA / pPa, KAPPA);
}

/** Saturation vapor pressure over liquid water, hPa. Magnus formula. tC in °C. */
export function satVaporPressureHPa(tC: number): number {
  return 6.112 * Math.exp((17.67 * tC) / (tC + 243.5));
}

/** Partial vapor pressure Pa: e = qv·p/(ε+qv). qv kg/kg, p Pa. */
export function partialVaporPressurePa(qv: number, pPa: number): number {
  if (qv <= 0 || !Number.isFinite(qv)) return 0;
  return (qv * pPa) / (EPSILON + qv);
}

/** Dewpoint °C from mixing ratio kg/kg + pressure Pa. Inverts Magnus. Returns NaN when qv ≤ 0. */
export function dewpointCelsiusFromQv(qv: number, pPa: number): number {
  const ePa = partialVaporPressurePa(qv, pPa);
  if (ePa <= 0) return NaN;
  const eHPa = ePa / 100;
  const x = Math.log(eHPa / 6.112);
  const denom = 17.67 - x;
  if (denom <= 0) return NaN;
  return (243.5 * x) / denom;
}

/**
 * Bolton (1980) pseudoequivalent potential temperature θe. Uses Bolton Eq. 15 (T_LCL) and Eq. 43 (θe).
 * qv is dry-air mixing ratio kg/kg (CM1/Nimbus convention); p total pressure Pa.
 */
export function equivalentPotentialTemperatureBolton(
  thetaK: number,
  pPa: number,
  qv: number,
): number {
  if (
    !(thetaK > 0)
    || !(pPa > 0)
    || qv < 0
    || !Number.isFinite(thetaK)
    || !Number.isFinite(pPa)
    || !Number.isFinite(qv)
  ) return NaN;

  const temperatureK = tempFromTheta(thetaK, pPa);
  if (!(temperatureK > 0) || !Number.isFinite(temperatureK)) return NaN;

  // Bolton Eq. 43 has a well-defined zero-mixing-ratio limit.
  let lclTemperatureK = temperatureK;
  if (qv > 0) {
    const dewpointK = dewpointCelsiusFromQv(qv, pPa) + T_ZERO;
    if (!(dewpointK > 56) || !Number.isFinite(dewpointK)) return NaN;
    const reciprocalLcl = 1 / (dewpointK - 56) + Math.log(temperatureK / dewpointK) / 800;
    if (!(reciprocalLcl > 0) || !Number.isFinite(reciprocalLcl)) return NaN;
    lclTemperatureK = 1 / reciprocalLcl + 56;
  }

  const mixingRatioGKg = qv * 1_000;
  const pressureHpa = pPa / 100;
  const dryPotentialTemperature = temperatureK * Math.pow(
    1_000 / pressureHpa,
    0.2854 * (1 - 0.00028 * mixingRatioGKg),
  );
  const moistureExponent = (3.376 / lclTemperatureK - 0.00254)
    * mixingRatioGKg
    * (1 + 0.00081 * mixingRatioGKg);
  const thetaE = dryPotentialTemperature * Math.exp(moistureExponent);
  return thetaE > 0 && Number.isFinite(thetaE) ? thetaE : NaN;
}

/** Temperature column °C: uses direct T K when fieldIsTemperature, else derives T from θ + p. */
export function columnTemperatureCelsius(
  thetaOrT: ArrayLike<number>,
  pPa: ArrayLike<number>,
  fieldIsTemperature: boolean,
): Float32Array {
  const n = Math.min(thetaOrT.length, pPa.length);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const tK = fieldIsTemperature
      ? thetaOrT[i]
      : tempFromTheta(thetaOrT[i], pPa[i]);
    out[i] = kelvinToCelsius(tK);
  }
  return out;
}

/** Dewpoint column °C from qv kg/kg + p Pa profiles. */
export function columnDewpointCelsius(
  qv: ArrayLike<number>,
  pPa: ArrayLike<number>,
): Float32Array {
  const n = Math.min(qv.length, pPa.length);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = dewpointCelsiusFromQv(qv[i], pPa[i]);
  }
  return out;
}

/* ---------- Skew-T transform ---------- */

/*
 * Skew-T display transform: y = log10(P_REF_HPA/p), x = T_C + skew·y.
 * skew = 30 °C/decade gives ~45° isotherms on a square plot.
 */
export const SKEW_C_PER_DECADE = 30;
export const P_REF_HPA = 1000;

export function skewX(tC: number, pHPa: number, skew = SKEW_C_PER_DECADE): number {
  return tC + skew * Math.log10(P_REF_HPA / pHPa);
}

export function pToY(pHPa: number): number {
  return Math.log10(P_REF_HPA / pHPa);
}

/* ---------- Reference curves for the plot background ---------- */

/** Dry adiabat (const θ): T(p) = θ·(p/p_ref)^κ. Returns T_C and p_hPa arrays uniform in log-p. */
export function dryAdiabat(
  thetaK: number,
  pTopHPa: number,
  pBottomHPa: number,
  samples = 32,
): { tC: number[]; pHPa: number[] } {
  const tC: number[] = new Array(samples);
  const pHPa: number[] = new Array(samples);
  // Uniform in log p to match the plot's log-y axis.
  const yTop = Math.log10(pTopHPa);
  const yBot = Math.log10(pBottomHPa);
  for (let i = 0; i < samples; i++) {
    const frac = i / (samples - 1);
    const yLog = yBot + frac * (yTop - yBot);
    const p = Math.pow(10, yLog);
    const tK = tempFromTheta(thetaK, p * 100);
    tC[i] = kelvinToCelsius(tK);
    pHPa[i] = p;
  }
  return { tC, pHPa };
}

/** Saturation mixing ratio g/kg: w_s = ε·e_s/(p−e_s). Returns NaN when p ≤ e_s. */
export function saturationMixingRatioGKg(tC: number, pHPa: number): number {
  const es = satVaporPressureHPa(tC); // hPa
  if (!Number.isFinite(es) || pHPa <= es) return NaN;
  return (1000 * (EPSILON * es)) / (pHPa - es);
}

/** Temperature °C where saturation mixing ratio equals wGKg at pHPa. Inverse of saturationMixingRatioGKg in T. */
export function temperatureForSaturationMixingRatio(
  wGKg: number,
  pHPa: number,
): number {
  const w = wGKg / 1000; // kg/kg
  if (w <= 0) return NaN;
  const es = (w * pHPa) / (EPSILON + w); // hPa
  if (es <= 0) return NaN;
  const x = Math.log(es / 6.112);
  const denom = 17.67 - x;
  if (denom <= 0) return NaN;
  return (243.5 * x) / denom;
}

/** Isohume (constant w) T_C(p) and p_hPa arrays, uniform in log-p. */
export function saturationMixingRatioLine(
  wGKg: number,
  pTopHPa: number,
  pBottomHPa: number,
  samples = 24,
): { tC: number[]; pHPa: number[] } {
  const tC: number[] = new Array(samples);
  const pHPa: number[] = new Array(samples);
  const yTop = Math.log10(pTopHPa);
  const yBot = Math.log10(pBottomHPa);
  for (let i = 0; i < samples; i++) {
    const frac = i / (samples - 1);
    const p = Math.pow(10, yBot + frac * (yTop - yBot));
    tC[i] = temperatureForSaturationMixingRatio(wGKg, p);
    pHPa[i] = p;
  }
  return { tC, pHPa };
}

/** Pseudoadiabatic lapse rate dT/d(ln p) = κ·T·(1 + L·w_s/(R_d·T)) / (1 + ε·L²·w_s/(c_p·R_d·T²)). T K, p hPa. */
function moistLapseDtDlnp(tK: number, pHPa: number): number {
  const tCval = tK - T_ZERO;
  const es = satVaporPressureHPa(tCval); // hPa
  const ws = pHPa > es ? (EPSILON * es) / (pHPa - es) : 0; // kg/kg
  const num = 1 + (L_V * ws) / (R_D * tK);
  const den = 1 + (EPSILON * L_V * L_V * ws) / (C_P * R_D * tK * tK);
  return KAPPA * tK * (num / den);
}

/** One RK4 step of the pseudoadiabat ODE from y0 to y1 (y = ln p_hPa). */
function moistRk4(tK: number, y0: number, y1: number): number {
  const h = y1 - y0;
  const f = (y: number, T: number) => moistLapseDtDlnp(T, Math.exp(y));
  const k1 = f(y0, tK);
  const k2 = f(y0 + h / 2, tK + (h / 2) * k1);
  const k3 = f(y0 + h / 2, tK + (h / 2) * k2);
  const k4 = f(y0 + h, tK + h * k3);
  return tK + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
}

/** Moist (pseudo) adiabat labelled by its 1000 hPa temperature (≈ θ_w). Returns T_C and p_hPa uniform in log-p. */
export function moistAdiabat(
  t1000C: number,
  pTopHPa: number,
  pBottomHPa: number,
  samples = 48,
): { tC: number[]; pHPa: number[] } {
  const tC: number[] = new Array(samples);
  const pHPa: number[] = new Array(samples);

  // Descending-pressure (bottom first), log-spaced.
  const yBot = Math.log(pBottomHPa);
  const yTop = Math.log(pTopHPa);
  const ys: number[] = new Array(samples);
  for (let i = 0; i < samples; i++) {
    ys[i] = yBot + (i / (samples - 1)) * (yTop - yBot);
    pHPa[i] = Math.exp(ys[i]);
  }

  // Seed from the 1000 hPa reference, then march sample-to-sample.
  const yRef = Math.log(1000);
  let tK = celsiusToKelvin(t1000C);
  tK = moistRk4(tK, yRef, ys[0]);
  tC[0] = kelvinToCelsius(tK);
  for (let i = 1; i < samples; i++) {
    tK = moistRk4(tK, ys[i - 1], ys[i]);
    tC[i] = kelvinToCelsius(tK);
  }
  return { tC, pHPa };
}

/* ---------- Lifted-parcel ascent + CAPE / CIN (storm diagnostics) ---------- */

/* ---------- Lifted-parcel ascent + CAPE / CIN (storm diagnostics) ---------- */

/** Standard gravity, m/s². Used by the buoyancy and hypsometric integrals. */
export const G_ACCEL = 9.81;

/** Saturation mixing ratio kg/kg: r_s = ε·e_s/(p−e_s). tK in K, pHPa in hPa. Returns 0 when p ≤ e_s. */
export function saturationMixingRatioForPressure(tK: number, pHPa: number): number {
  const es = satVaporPressureHPa(tK - T_ZERO); // hPa
  if (!Number.isFinite(es) || pHPa <= es) return 0;
  return (EPSILON * es) / (pHPa - es);
}

/** Virtual temperature K: T_v = T·(1 + 0.61·r). Emanuel (1994); r < 0 or non-finite treated as dry. */
export function virtualTemperatureK(tK: number, r: number): number {
  const rr = Number.isFinite(r) && r > 0 ? r : 0;
  return tK * (1 + 0.61 * rr);
}

/**
 * LCL via Bolton (1980) Eq. 15: T_LCL = 1/(1/(T_d−56) + ln(T/T_d)/800) + 56 [K].
 * p_LCL = p·(T_LCL/T)^(c_p/R_d). Dry parcels (qv ≤ 0) return the input level.
 */
export function liftingCondensationLevel(
  tK: number,
  pHPa: number,
  qv: number,
): { T_LCL_K: number; p_LCL_hPa: number } {
  if (!(qv > 0) || !Number.isFinite(qv)) {
    return { T_LCL_K: tK, p_LCL_hPa: pHPa };
  }
  const tdK = dewpointCelsiusFromQv(qv, pHPa * 100) + T_ZERO;
  const T_LCL_K = 1 / (1 / (tdK - 56) + Math.log(tK / tdK) / 800) + 56;
  const p_LCL_hPa = pHPa * Math.pow(T_LCL_K / tK, C_P / R_D);
  return { T_LCL_K, p_LCL_hPa };
}

/** Lifted-parcel temperature path, surface (highest p) first. r_kg_per_kg: conserved below LCL, saturation above. */
export interface ParcelPath {
  p_hPa: number[];
  T_K: number[];
  T_C: number[];
  r_kg_per_kg: number[];
  LCL_hPa: number;
}

/**
 * Lift a surface parcel from p_sfc_hPa to p_top_hPa. Dry adiabat below the LCL
 * (qv conserved); pseudo-adiabat above (RK4 stepper). LCL inserted as an exact node.
 */
export function parcelAscent(
  T_sfc_K: number,
  p_sfc_hPa: number,
  qv_sfc: number,
  p_top_hPa: number,
  n_steps = 150,
): ParcelPath {
  const hasMoisture = Number.isFinite(qv_sfc) && qv_sfc > 0;
  const { T_LCL_K, p_LCL_hPa } = liftingCondensationLevel(
    T_sfc_K,
    p_sfc_hPa,
    qv_sfc,
  );
  const thetaParcel = thetaFromTemp(T_sfc_K, p_sfc_hPa * 100);

  // Degenerate / invalid columns: return just the surface node.
  if (!(p_top_hPa < p_sfc_hPa) || !Number.isFinite(p_top_hPa)) {
    return {
      p_hPa: [p_sfc_hPa],
      T_K: [T_sfc_K],
      T_C: [kelvinToCelsius(T_sfc_K)],
      r_kg_per_kg: [hasMoisture ? qv_sfc : 0],
      LCL_hPa: hasMoisture ? p_LCL_hPa : p_sfc_hPa,
    };
  }

  // Descending-pressure nodes, log-spaced, LCL inserted when inside the column.
  const steps = Math.max(2, Math.floor(n_steps));
  const yBot = Math.log(p_sfc_hPa);
  const yTop = Math.log(p_top_hPa);
  const pressures: number[] = [];
  for (let i = 0; i < steps; i++) {
    pressures.push(Math.exp(yBot + (i / (steps - 1)) * (yTop - yBot)));
  }
  if (hasMoisture && p_LCL_hPa < p_sfc_hPa && p_LCL_hPa > p_top_hPa) {
    pressures.push(p_LCL_hPa);
    pressures.sort((a, b) => b - a); // keep descending
  }

  const p_hPa: number[] = [];
  const T_K: number[] = [];
  const T_C: number[] = [];
  const r_kg_per_kg: number[] = [];

  // March the moist segment node-to-node from the LCL using the shared RK4 stepper.
  let moistTK = T_LCL_K;
  let moistY = Math.log(p_LCL_hPa);
  let moistSeeded = false;

  for (let i = 0; i < pressures.length; i++) {
    const p = pressures[i];
    if (i > 0 && Math.abs(p - pressures[i - 1]) < 1e-9) continue; // dedupe LCL
    let tK: number;
    let r: number;
    if (!hasMoisture || p >= p_LCL_hPa) {
      tK = tempFromTheta(thetaParcel, p * 100);
      r = hasMoisture ? qv_sfc : 0;
    } else {
      // Pseudo-adiabat above the LCL.
      if (!moistSeeded) {
        moistTK = T_LCL_K;
        moistY = Math.log(p_LCL_hPa);
        moistSeeded = true;
      }
      const y = Math.log(p);
      moistTK = moistRk4(moistTK, moistY, y);
      moistY = y;
      tK = moistTK;
      r = saturationMixingRatioForPressure(tK, p);
    }
    p_hPa.push(p);
    T_K.push(tK);
    T_C.push(kelvinToCelsius(tK));
    r_kg_per_kg.push(r);
  }

  return {
    p_hPa,
    T_K,
    T_C,
    r_kg_per_kg,
    LCL_hPa: hasMoisture ? p_LCL_hPa : p_sfc_hPa,
  };
}

/** CAPE/CIN and the parcel's LFC/EL, in display units. */
export interface ParcelDiagnostics {
  CAPE_J_per_kg: number;
  CIN_J_per_kg: number;
  LFC_hPa: number | null;
  EL_hPa: number | null;
}

/**
 * CAPE/CIN and LFC/EL for a lifted parcel vs environment. Buoyancy uses virtual temperature
 * B = g·(T_v,parcel − T_v,env)/T_v,env; heights from hypsometric equation; trapezoidal integrals.
 * LFC/EL bound the LARGEST positive area. CAPE J/kg, CIN J/kg (positive).
 */
export function parcelDiagnostics(
  path: ParcelPath,
  envTempK: (pHPa: number) => number,
  envQv: (pHPa: number) => number,
): ParcelDiagnostics {
  const n = path.p_hPa.length;
  const none: ParcelDiagnostics = {
    CAPE_J_per_kg: 0,
    CIN_J_per_kg: 0,
    LFC_hPa: null,
    EL_hPa: null,
  };
  if (n < 2) return none;

  // Per-level buoyancy and cumulative height (surface z = 0).
  const z: number[] = new Array(n);
  const B: number[] = new Array(n);
  const tvEnv: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const p = path.p_hPa[i];
    tvEnv[i] = virtualTemperatureK(envTempK(p), envQv(p));
    const tvParcel = virtualTemperatureK(path.T_K[i], path.r_kg_per_kg[i]);
    B[i] = (G_ACCEL * (tvParcel - tvEnv[i])) / tvEnv[i];
  }
  z[0] = 0;
  for (let i = 1; i < n; i++) {
    const pLower = path.p_hPa[i - 1];
    const pUpper = path.p_hPa[i];
    const tvMean = 0.5 * (tvEnv[i - 1] + tvEnv[i]);
    z[i] = z[i - 1] + (R_D / G_ACCEL) * tvMean * Math.log(pLower / pUpper);
  }

  // Refine into sign-pure segments by inserting B = 0 nodes at zero-crossings.
  const Z: number[] = [z[0]];
  const Bv: number[] = [B[0]];
  const P: number[] = [path.p_hPa[0]];
  for (let i = 0; i < n - 1; i++) {
    if (B[i] * B[i + 1] < 0) {
      const f = B[i] / (B[i] - B[i + 1]); // fraction to the zero
      Z.push(z[i] + f * (z[i + 1] - z[i]));
      Bv.push(0);
      // Log-linear in p, because z above is hypsometric (linear in ln p). Interpolating p linearly
      // against the same fraction reports an LFC/EL a few hPa off the height the crossing was at.
      P.push(Math.exp(
        Math.log(path.p_hPa[i]) + f * (Math.log(path.p_hPa[i + 1]) - Math.log(path.p_hPa[i])),
      ));
    }
    Z.push(z[i + 1]);
    Bv.push(B[i + 1]);
    P.push(path.p_hPa[i + 1]);
  }

  // LFC/EL: pair bounding the largest positive area (not first) to avoid false shallow-bubble
  // anchoring. CAPE is therefore that layer alone, NOT the sum of every positive layer: a sounding
  // with elevated instability above a stable layer reads lower here than standard SBCAPE. Callers
  // must label it as dominant-layer CAPE rather than plain CAPE.
  let cape = 0;
  let lfcIdx = -1;
  let elIdx = -1;
  for (let i = 0; i < Bv.length; ) {
    if (!(Bv[i] > 0)) {
      i++;
      continue;
    }
    // Maximal run of positive nodes [i, top).
    let top = i;
    while (top < Bv.length && Bv[top] > 0) top++;
    const start = i === 0 ? 0 : i - 1;
    const restabilizes = top < Bv.length;
    const end = restabilizes ? top : Bv.length - 1;
    let area = 0;
    for (let k = start; k < end; k++) {
      area += 0.5 * (Bv[k] + Bv[k + 1]) * (Z[k + 1] - Z[k]);
    }
    if (area > cape) {
      cape = area;
      lfcIdx = start;
      elIdx = restabilizes ? end : -1;
    }
    i = top + 1;
  }
  if (lfcIdx === -1) return none; // never buoyant -> no CAPE, no LFC/EL

  // CIN: negative area below LFC only (segments are sign-pure after zero-crossing insertion).
  let cin = 0;
  for (let i = 0; i < lfcIdx; i++) {
    if (Bv[i] <= 0 && Bv[i + 1] <= 0) {
      cin += 0.5 * (Bv[i] + Bv[i + 1]) * (Z[i + 1] - Z[i]);
    }
  }

  return {
    CAPE_J_per_kg: cape,
    CIN_J_per_kg: -cin, // negative area below the LFC, reported positive
    LFC_hPa: P[lfcIdx],
    EL_hPa: elIdx === -1 ? null : P[elIdx],
  };
}
