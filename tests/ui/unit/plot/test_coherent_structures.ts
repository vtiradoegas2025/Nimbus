/*
 * Tests for `src/lib/plot/coherentStructures.ts` — which fields carry a documented isosurface
 * value, and which are left to be derived from the data.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types --import ../tests/ui/register-ts-resolve.mjs ../tests/ui/unit/plot/test_coherent_structures.ts
 */

import {
  physicalIsosurfaceThreshold,
  defaultsToSignedPair,
  defaultsToPositiveQSurface,
  defaultsToRotationDominatedSurface,
  isosurfaceWindowAggregation,
  isosurfaceWindowAggregations,
} from "../../../../ui/src/lib/plot/coherentStructures.ts";

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

/** 0.1 g/kg in kg/kg — the conventional condensate-presence value. */
const PRESENCE = 1.0e-4;

const CONDENSATE = [
  "qc", "qr", "qi", "qs", "qg", "qh",
  "cloud_condensate", "total_condensate",
  "cloud_liquid_condensate", "cloud_frozen_condensate",
];

// ---------- Condensate ----------

check("every condensate species opens at the presence threshold", () => {
  /*
   * Condensate is sparse: most of a domain holds none, so a range-derived default lands
   * near the peak and draws the densest core instead of the cloud body. Measured on a real
   * beast_xl frame, cloud_condensate's range-derived value sat above its own p99 while the
   * rendered-supercell widget called the same field cloud at 1e-4 — a 58x disagreement
   * between two views of one quantity.
   */
  for (const field of CONDENSATE) {
    assertTrue(physicalIsosurfaceThreshold(field) === PRESENCE, `${field} opens at 0.1 g/kg`);
  }
});

check("water vapour is deliberately excluded", () => {
  // qv is present throughout the domain, so a range-derived default describes it well.
  // Pinning it at the condensate value would draw a surface through most of the volume.
  assertTrue(physicalIsosurfaceThreshold("qv") === undefined, "qv has no fixed default");
});

// ---------- Dynamics ----------

check("the documented dynamics defaults are unchanged", () => {
  assertTrue(physicalIsosurfaceThreshold("w") === 10.0, "w opens at 10 m/s");
  assertTrue(physicalIsosurfaceThreshold("vorticity_magnitude") === 0.1, "|omega| opens at 0.1 /s");
  assertTrue(physicalIsosurfaceThreshold("vorticity_z") === 0.1, "zeta opens at 0.1 /s");
});

check("fields with no documented value derive one from the data", () => {
  for (const field of ["theta_prime", "buoyancy", "p_prime", "reflectivity_dbz", "u", "v"]) {
    assertTrue(physicalIsosurfaceThreshold(field) === undefined, `${field} is derived`);
  }
});

check("a missing field name is handled rather than thrown on", () => {
  assertTrue(physicalIsosurfaceThreshold(null) === undefined, "null yields no default");
  assertTrue(physicalIsosurfaceThreshold("") === undefined, "empty name yields no default");
});

check("the lookup does not inherit from Object.prototype", () => {
  // A bare Record lookup answers "constructor" and "toString" with functions; the widget
  // would then set a function as an isosurface value.
  for (const key of ["constructor", "toString", "hasOwnProperty", "__proto__"]) {
    const value = physicalIsosurfaceThreshold(key);
    assertTrue(
      value === undefined || typeof value === "number",
      `${key} does not leak a prototype member`,
    );
  }
});

// ---------- Interaction with the surface-shape predicates ----------

check("condensate is not treated as a signed pair", () => {
  // A mixing ratio is non-negative; a paired positive/negative surface has no meaning.
  for (const field of CONDENSATE) {
    assertTrue(!defaultsToSignedPair(field), `${field} is not a signed pair`);
    assertTrue(!defaultsToPositiveQSurface(field), `${field} is not the Q-criterion surface`);
    assertTrue(!defaultsToRotationDominatedSurface(field), `${field} is not rotation-dominated`);
  }
});

check("the signed-pair fields still are", () => {
  for (const field of ["vorticity_z", "stretching_term", "tilting_term", "baroclinic_term"]) {
    assertTrue(defaultsToSignedPair(field), `${field} defaults to a signed pair`);
  }
});

check("isosurface LOD preserves threshold extrema instead of averaging them away", () => {
  assertTrue(isosurfaceWindowAggregation("w") === "block_max", "positive updraft peaks use block max");
  assertTrue(isosurfaceWindowAggregation("reflectivity_dbz") === "block_max", "reflectivity cores use block max");
  assertTrue(isosurfaceWindowAggregation("okubo_weiss") === "block_min", "rotation-dominated minima use block min");
  assertTrue(isosurfaceWindowAggregation("vorticity_z") === "magnitude_extreme", "signed rotation keeps its strongest sense");
  assertTrue(
    isosurfaceWindowAggregations("vorticity_z", true).join(",") === "block_max,block_min",
    "reduced signed surfaces keep independent positive and negative extrema",
  );
  assertTrue(
    isosurfaceWindowAggregations("vorticity_z", false).join(",") === "magnitude_extreme",
    "resolved signed surfaces retain their exact single lattice",
  );
});

// ---------- Summary ----------

for (const line of failures) console.error(line);
console.log(`\ncoherent-structures tests: ${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
