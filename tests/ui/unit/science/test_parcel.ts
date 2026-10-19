/*
 * Tests for the lifted-parcel + CAPE/CIN diagnostics in `src/lib/thermo.ts`.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types scripts/test_parcel.ts
 *
 * Assertions reference EXTERNAL TRUTH, not the implementation:
 *   - LCL is checked against Bolton (1980) eq. (15) AND an independent
 *     cross-check using Bolton's eq. (22) (vapor-pressure form) plus the
 *     physical Espy estimate.
 *   - CAPE/CIN/LFC/EL are checked against a synthetic isothermal-environment
 *     sounding whose buoyancy is piecewise-linear in height, so the trapezoidal
 *     integral is hand-computable exactly.
 *
 * NOTE on the task prompt: it states the (T=30 C, Td=20 C) parcel lifts to
 * T_LCL ~ 288.1 K (15.0 C). That figure is incorrect. Bolton eq. (15), Bolton
 * eq. (22), and the Espy height approximation all agree the LCL temperature is
 * ~290.8 K (~17.7 C), with p_LCL ~ 864.7 hPa from p=1000 hPa. The assertions
 * below use the correct ~290.8 K value; matching 288.1 K would require
 * corrupting the physics.
 */

import {
  C_P,
  EPSILON,
  liftingCondensationLevel,
  parcelAscent,
  parcelDiagnostics,
  pressureUnitFromLabel,
  R_D,
  saturationMixingRatioForPressure,
  T_ZERO,
  virtualTemperatureK,
  type ParcelPath,
} from "../../../../ui/src/lib/science/thermo.ts";

// ---------- Tiny assertion API (matches test_diagnostics.ts) ----------

let failed = 0;
let passed = 0;
const failures: string[] = [];

function assertTrue(cond: boolean, label: string): void {
  if (cond) {
    passed++;
  } else {
    failed++;
    failures.push(`FAIL ${label}`);
  }
}

function assertClose(
  actual: number,
  expected: number,
  tol: number,
  label: string,
): void {
  const ok = Number.isFinite(actual) && Math.abs(actual - expected) <= tol;
  if (ok) {
    passed++;
  } else {
    failed++;
    failures.push(
      `FAIL ${label}\n        expected: ${expected} +/- ${tol}\n        actual:   ${actual}`,
    );
  }
}

function check(label: string, fn: () => void): void {
  try {
    fn();
  } catch (e) {
    failed++;
    failures.push(`THREW ${label}: ${(e as Error).message}`);
  }
}

check("manifest pressure-unit labels resolve without inspecting data magnitude", () => {
  assertTrue(pressureUnitFromLabel("Pa") === "Pa", "Pa catalog label");
  assertTrue(pressureUnitFromLabel("pascals") === "Pa", "pascal alias");
  assertTrue(pressureUnitFromLabel("hPa") === "hPa", "hPa catalog label");
  assertTrue(pressureUnitFromLabel("mb") === "hPa", "millibar alias");
  assertTrue(pressureUnitFromLabel("psi") === null, "unsupported unit rejected");
  assertTrue(pressureUnitFromLabel(null) === null, "missing legacy unit remains unknown");
});

// ---------- 1. LCL vs Bolton (1980) ----------

check("LCL matches Bolton (1980) eq. (15) for T=30C, Td=20C", () => {
  const T_sfc_K = 30 + T_ZERO;
  const p_sfc_hPa = 1000;
  // Mixing ratio of a parcel with dewpoint 20 C at 1000 hPa: the actual vapor
  // mixing ratio equals the SATURATION mixing ratio evaluated at the dewpoint.
  const qv = saturationMixingRatioForPressure(20 + T_ZERO, p_sfc_hPa);

  const { T_LCL_K, p_LCL_hPa } = liftingCondensationLevel(T_sfc_K, p_sfc_hPa, qv);

  // Independent oracle: Bolton eq. (22), the vapor-pressure form (algebraically
  // distinct from eq. (15), which the implementation uses).
  const e_hPa = (qv * p_sfc_hPa) / (EPSILON + qv);
  const T_LCL_22 =
    2840 / (3.5 * Math.log(T_sfc_K) - Math.log(e_hPa) - 4.805) + 55;
  const p_LCL_oracle = p_sfc_hPa * Math.pow(T_LCL_22 / T_sfc_K, C_P / R_D);

  console.log(
    `  LCL: impl T_LCL=${T_LCL_K.toFixed(3)} K (${(T_LCL_K - T_ZERO).toFixed(2)} C), ` +
      `p_LCL=${p_LCL_hPa.toFixed(2)} hPa; eq22 oracle T_LCL=${T_LCL_22.toFixed(3)} K`,
  );

  // Cross-check against the independent eq. (22) oracle.
  assertClose(T_LCL_K, T_LCL_22, 0.1, "T_LCL agrees with Bolton eq. (22)");
  assertClose(p_LCL_hPa, p_LCL_oracle, 2, "p_LCL agrees with Poisson(eq22 T_LCL)");

  // Pin to the literal published / physical values (NOT the prompt's 288.1 K).
  assertClose(T_LCL_K, 290.8, 0.3, "T_LCL ~ 290.8 K (correct Bolton value)");
  assertClose(p_LCL_hPa, 864.7, 2, "p_LCL ~ 864.7 hPa (hand Poisson)");
});

// ---------- 2. CAPE on an analytical sounding ----------
//
// Environment: isothermal and dry, T_v,env = 250 K everywhere. With the
// hypsometric scale (R_d/g)*250 = 7315.2 m, heights are z = 7315.2*ln(1000/p).
// The parcel is dry (r = 0 -> T_v = T), so B = g*(T_p - 250)/250 = 0.03924*dT.
//
//   p(hPa)  dT(K)   z(m)        B(m/s^2)
//   1000    -2      0           -0.07848
//    900     0      770.74       0          <- LFC (zero-crossing on a node)
//    700    +11     2609.16     +0.43164
//    400    +11     6702.93     +0.43164
//    250     0     10141.13      0          <- EL
//
// Trapezoidal CAPE (B piecewise-linear in z, so this is exact):
//   900->700 : 0.5*0.43164*1838.42 = 396.8
//   700->400 :     0.43164*4093.77 = 1767.0
//   400->250 : 0.5*0.43164*3438.20 =  742.1   => CAPE ~ 2906 J/kg
// CIN (below LFC): -0.5*(-0.07848)*770.74 = 30.2 J/kg.

check("CAPE/CIN/LFC/EL on a hand-computable isothermal sounding", () => {
  const T_K = [248, 250, 261, 261, 250];
  const path: ParcelPath = {
    p_hPa: [1000, 900, 700, 400, 250],
    T_K,
    T_C: T_K.map((t) => t - T_ZERO),
    r_kg_per_kg: [0, 0, 0, 0, 0],
    LCL_hPa: 900,
  };
  const diag = parcelDiagnostics(
    path,
    () => 250, // isothermal environment
    () => 0, // dry environment
  );

  console.log(
    `  CAPE case: CAPE=${diag.CAPE_J_per_kg.toFixed(1)} J/kg, ` +
      `CIN=${diag.CIN_J_per_kg.toFixed(1)} J/kg, ` +
      `LFC=${diag.LFC_hPa} hPa, EL=${diag.EL_hPa} hPa`,
  );

  assertClose(diag.CAPE_J_per_kg, 2906, 2906 * 0.05, "CAPE within 5% of 2906 J/kg");
  assertClose(diag.CIN_J_per_kg, 30.2, 10, "CIN within 10 J/kg of 30.2");
  assertTrue(diag.LFC_hPa !== null, "LFC found");
  assertTrue(diag.EL_hPa !== null, "EL found");
  if (diag.LFC_hPa !== null) assertClose(diag.LFC_hPa, 900, 5, "LFC ~ 900 hPa");
  if (diag.EL_hPa !== null) assertClose(diag.EL_hPa, 250, 5, "EL ~ 250 hPa");
});

// ---------- 2b. Shallow surface bubble must not steal the LFC ----------
//
// REGRESSION: selecting the FIRST positive node as the LFC anchors it inside a
// superadiabatic surface layer, then closes the EL at the top of that bubble —
// throwing away the whole deep positive area above the cap. The correct answer
// is the pair bounding the DEEPEST positive area.
//
// Same isothermal environment (T_v,env = 250 K, z = 7315.2*ln(1000/p)) as case
// 2, with a shallow buoyant bubble spliced in below a capping inversion:
//
//   p(hPa)  dT(K)   z(m)        B(m/s^2)
//   1000     0      0            0
//    980    +1      147.85      +0.03924   <- shallow surface bubble
//    960     0      298.65       0
//    900    -2      770.74      -0.07848   <- the cap
//    850     0      1177.34      0         <- true LFC
//    700    +11     2609.16     +0.43164
//    400    +11     6702.93     +0.43164
//    250     0     10141.13      0         <- true EL
//
// Bubble area  = 0.5*0.03924*298.65 =    5.9 J/kg  (must LOSE)
// Deep area    = 0.5*0.43164*1431.82
//              +     0.43164*4093.77
//              + 0.5*0.43164*3438.20 = 2818.1 J/kg  (must WIN)
// CIN (negative area below 850 hPa, both sides of the cap):
//   960->900 : 0.5*(-0.07848)*472.09 = -18.5
//   900->850 : 0.5*(-0.07848)*406.60 = -16.0   => CIN ~ 34.5 J/kg

check("a shallow surface bubble does not capture the LFC from the deep CAPE", () => {
  const T_K = [250, 251, 250, 248, 250, 261, 261, 250];
  const path: ParcelPath = {
    p_hPa: [1000, 980, 960, 900, 850, 700, 400, 250],
    T_K,
    T_C: T_K.map((t) => t - T_ZERO),
    r_kg_per_kg: new Array(8).fill(0),
    LCL_hPa: 850,
  };
  const diag = parcelDiagnostics(path, () => 250, () => 0);

  console.log(
    `  bubble case: CAPE=${diag.CAPE_J_per_kg.toFixed(1)} J/kg, ` +
      `CIN=${diag.CIN_J_per_kg.toFixed(1)} J/kg, ` +
      `LFC=${diag.LFC_hPa} hPa, EL=${diag.EL_hPa} hPa`,
  );

  assertClose(diag.CAPE_J_per_kg, 2818, 2818 * 0.05, "CAPE is the deep area, not the bubble");
  assertTrue(diag.CAPE_J_per_kg > 100, "CAPE did not collapse to the surface bubble");
  assertClose(diag.CIN_J_per_kg, 34.5, 10, "CIN within 10 J/kg of 34.5");
  assertTrue(diag.CIN_J_per_kg > 0, "CIN is not zeroed by an LFC at the surface");
  if (diag.LFC_hPa !== null) assertClose(diag.LFC_hPa, 850, 5, "LFC ~ 850 hPa (above the cap)");
  if (diag.EL_hPa !== null) assertClose(diag.EL_hPa, 250, 5, "EL ~ 250 hPa (top of the deep area)");
});

// ---------- 3. Stable sounding -> zero CAPE ----------

check("monotonically stable environment gives zero CAPE and no LFC/EL", () => {
  // Parcel everywhere COOLER than the environment -> B < 0 at every level.
  const T_K = [290, 280, 270, 260, 250];
  const path: ParcelPath = {
    p_hPa: [1000, 900, 700, 400, 250],
    T_K,
    T_C: T_K.map((t) => t - T_ZERO),
    r_kg_per_kg: [0, 0, 0, 0, 0],
    LCL_hPa: 1000,
  };
  const diag = parcelDiagnostics(
    path,
    // Warming-with-height environment (T rises as p falls); it stays warmer
    // than the parcel at every level, so the parcel is never buoyant.
    (p) => 300 + 10 * Math.log(1000 / p),
    () => 0,
  );

  assertTrue(diag.CAPE_J_per_kg === 0, "CAPE is exactly 0 for a stable sounding");
  assertTrue(diag.LFC_hPa === null, "no LFC when never buoyant");
  assertTrue(diag.EL_hPa === null, "no EL when never buoyant");
});

// ---------- 4. Saturated surface parcel -> p_LCL ~ p_sfc ----------

check("saturated surface parcel has its LCL at the surface", () => {
  const p_sfc_hPa = 1000;
  const T_sfc_K = 20 + T_ZERO;
  // T == Td: the surface mixing ratio is the saturation value at (T, p).
  const qv = saturationMixingRatioForPressure(T_sfc_K, p_sfc_hPa);
  const { T_LCL_K, p_LCL_hPa } = liftingCondensationLevel(T_sfc_K, p_sfc_hPa, qv);
  assertClose(p_LCL_hPa, p_sfc_hPa, 5, "p_LCL within 5 hPa of p_sfc");
  assertClose(T_LCL_K, T_sfc_K, 0.5, "T_LCL ~ surface T for a saturated parcel");
});

// ---------- 5. Determinism ----------

check("parcelAscent + parcelDiagnostics are deterministic", () => {
  const qv = saturationMixingRatioForPressure(18 + T_ZERO, 1000);
  const a = parcelAscent(28 + T_ZERO, 1000, qv, 150);
  const b = parcelAscent(28 + T_ZERO, 1000, qv, 150);
  let identical = a.p_hPa.length === b.p_hPa.length && a.LCL_hPa === b.LCL_hPa;
  for (let i = 0; i < a.p_hPa.length && identical; i++) {
    if (a.T_K[i] !== b.T_K[i] || a.p_hPa[i] !== b.p_hPa[i]) identical = false;
  }
  assertTrue(identical, "identical parcel paths from identical inputs");

  const envT = (p: number) => 295 - 50 * Math.log(1000 / p);
  const envQ = () => 0.002;
  const d1 = parcelDiagnostics(a, envT, envQ);
  const d2 = parcelDiagnostics(b, envT, envQ);
  assertTrue(
    d1.CAPE_J_per_kg === d2.CAPE_J_per_kg && d1.CIN_J_per_kg === d2.CIN_J_per_kg,
    "identical CAPE/CIN from identical inputs",
  );
});

// ---------- 6. Virtual-temperature correction is applied ----------

check("buoyancy uses virtual temperature (humid parcel is more buoyant)", () => {
  // virtualTemperatureK closed form.
  assertClose(virtualTemperatureK(300, 0.02), 300 * 1.0122, 1e-6, "Tv = T(1+0.61 r)");

  // Same temperature in parcel and environment, but the parcel is moister:
  // dry-T buoyancy would be exactly 0; the Tv correction makes it positive.
  const T_K = [300, 295, 290];
  const path: ParcelPath = {
    p_hPa: [1000, 850, 700],
    T_K,
    T_C: T_K.map((t) => t - T_ZERO),
    r_kg_per_kg: [0.018, 0.014, 0.011], // humid parcel
    LCL_hPa: 1000,
  };
  // Environment has the SAME temperature but is dry.
  const envT = (p: number) => T_K[path.p_hPa.indexOf(p)] ?? 295;
  const diag = parcelDiagnostics(path, envT, () => 0);
  assertTrue(
    diag.CAPE_J_per_kg > 0,
    "moist parcel is buoyant vs an equally-warm dry environment",
  );
});

// ---------- Report ----------

console.log(`\nparcel tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  " + f);
  process.exit(1);
}
process.exit(0);
