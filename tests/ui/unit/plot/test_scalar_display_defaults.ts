/*
 * Tests for `src/lib/plot/scalarDisplayDefaults.ts` — which colormap and range a scalar field is
 * drawn with when nothing has been chosen for it.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types --import ../tests/ui/register-ts-resolve.mjs ../tests/ui/unit/plot/test_scalar_display_defaults.ts
 */

import { defaultScalarDisplay } from "../../../../ui/src/lib/plot/scalarDisplayDefaults.ts";

// ---------- Tiny assertion API (matches test_pane_layout.ts) ----------

let failed = 0;
let passed = 0;
const failures: string[] = [];

function assertTrue(cond: boolean, label: string): void {
  if (cond) passed++;
  else {
    failed++;
    failures.push(`FAIL ${label}`);
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

/** Units as the run manifest carries them. */
const UNITS: Record<string, string> = {
  reflectivity_dbz: "dBZ",
  composite_reflectivity: "dBZ",
  radar: "mm^6/m^3",
  vil: "kg/m^2",
  w: "m/s",
  theta: "K",
  qv: "kg/kg",
  vorticity_z: "s^-1",
  p_prime: "Pa",
  stp: "1",
};

// ---------- Reflectivity owns the anchored palette ----------

check("a dBZ field gets the anchored NWS ramp", () => {
  for (const field of ["reflectivity_dbz", "composite_reflectivity"]) {
    const display = defaultScalarDisplay(field, UNITS);
    assertTrue(display.colormap === "dBZ", `${field} -> dBZ palette`);
    assertTrue(display.rangeMode === "dBZ", `${field} -> dBZ range`);
  }
});

check("the unit decides, not the name", () => {
  // `radar` is linear reflectivity in mm^6/m^3 and must NOT take the dBZ ramp.
  const display = defaultScalarDisplay("radar", UNITS);
  assertTrue(display.colormap !== "dBZ", "linear radar is not dBZ-anchored");
  assertTrue(display.rangeMode !== "dBZ", "linear radar keeps a fitted range");
});

/*
 * The defect this rule exists to prevent.
 *
 * The dBZ ramp anchors its colours AT reflectivity values, so its colorbar is locked to [-30, 75]
 * whatever the data is. Carrying it over from a radar preset to another field relabelled the bar
 * with the new field's unit while the numbers on it still meant reflectivity.
 */
check("no non-reflectivity field ever takes the anchored palette", () => {
  for (const field of ["vil", "w", "theta", "qv", "vorticity_z", "p_prime", "stp"]) {
    const display = defaultScalarDisplay(field, UNITS);
    assertTrue(display.colormap !== "dBZ", `${field} must not inherit the dBZ palette`);
    assertTrue(display.rangeMode !== "dBZ", `${field} must not inherit the dBZ range`);
  }
});

// ---------- Signed fields centre on zero ----------

check("signed quantities get a diverging ramp on a symmetric range", () => {
  for (const field of [
    "u", "v", "w", "u10", "v10", "surface_stress_u", "surface_stress_v",
    "surface_sensible_heat_flux", "surface_latent_heat_flux", "surface_moisture_flux",
    "vorticity_z", "p_prime", "theta_prime", "divergence", "tilting_term",
  ]) {
    const display = defaultScalarDisplay(field, UNITS);
    assertTrue(display.colormap === "RdBu", `${field} -> diverging`);
    assertTrue(display.rangeMode === "symmetric", `${field} -> symmetric`);
  }
});

check("a strictly positive field is not centred on zero", () => {
  // Half a symmetric ramp is wasted on values that cannot occur.
  for (const field of ["theta", "qv", "vil"]) {
    const display = defaultScalarDisplay(field, UNITS);
    assertTrue(display.rangeMode === "auto", `${field} -> fitted range`);
    assertTrue(display.colormap === "Viridis", `${field} -> sequential ramp`);
  }
});

// ---------- Degenerate inputs ----------

check("no field selected falls back to a neutral default", () => {
  const display = defaultScalarDisplay(null, UNITS);
  assertTrue(display.colormap === "Viridis", "null field -> Viridis");
  assertTrue(display.rangeMode === "auto", "null field -> auto");
});

check("a missing units map still classifies by name", () => {
  // Without a manifest the name heuristic is all there is; it must not crash or
  // hand the anchored palette to something that is not reflectivity.
  assertTrue(defaultScalarDisplay("w", undefined).colormap === "RdBu", "w -> diverging with no units");
  assertTrue(defaultScalarDisplay("theta", null).colormap === "Viridis", "theta -> sequential with no units");
  assertTrue(
    defaultScalarDisplay("reflectivity_dbz", null).colormap === "dBZ",
    "name heuristic still catches reflectivity",
  );
});

check("a magnitude never gets a diverging ramp", () => {
  /*
   * A magnitude is non-negative by construction, so centring a diverging ramp on zero
   * spends half the colour range on values that cannot occur. Measured on a beast_xl frame,
   * vorticity_magnitude ran 0 .. 0.396 while being drawn on a +/-0.396 RdBu ramp.
   */
  for (const field of ["vorticity_magnitude", "horizontal_wind_speed", "wind_speed"]) {
    const display = defaultScalarDisplay(field, UNITS);
    assertTrue(display.colormap === "Viridis", `${field} -> sequential`);
    assertTrue(display.rangeMode === "auto", `${field} -> auto range`);
  }
});

check("the signed quantity it is derived from still diverges", () => {
  // The magnitude guard must not swallow the signed parent field.
  assertTrue(defaultScalarDisplay("vorticity_z", UNITS).colormap === "RdBu", "vorticity_z diverges");
  assertTrue(defaultScalarDisplay("vorticity_x", UNITS).colormap === "RdBu", "vorticity_x diverges");
});

check("helicity and buoyancy diverge about zero", () => {
  // Helicity's sign is its rotation sense: cyclonic against anticyclonic. On a sequential
  // ramp the sign change is invisible, which is the one thing the field is read for.
  for (const field of ["srh_0_1km", "srh_0_3km", "updraft_helicity", "helicity_density", "buoyancy"]) {
    const display = defaultScalarDisplay(field, UNITS);
    assertTrue(display.colormap === "RdBu", `${field} -> diverging`);
    assertTrue(display.rangeMode === "symmetric", `${field} -> symmetric range`);
  }
});

check("single-signed thermodynamics stays sequential", () => {
  // These have no meaningful zero crossing in a simulated column; a diverging ramp would
  // imply one. CIN is negative throughout rather than signed.
  for (const field of ["cape", "cin", "theta", "temperature", "rho", "qv", "qr", "vil", "lcl"]) {
    assertTrue(defaultScalarDisplay(field, UNITS).colormap === "Viridis", `${field} -> sequential`);
  }
});

check("the rule is deterministic", () => {
  const first = defaultScalarDisplay("w", UNITS);
  const second = defaultScalarDisplay("w", UNITS);
  assertTrue(first.colormap === second.colormap && first.rangeMode === second.rangeMode, "same input, same answer");
});

// ---------- Summary ----------

for (const line of failures) console.error(line);
console.log(`\nscalar-display-defaults tests: ${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
