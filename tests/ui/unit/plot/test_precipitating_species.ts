/*
 * Precipitating-species removal for `src/lib/plot/precipitatingSpecies.ts`.
 *
 * The bug this guards: the lumped-condensate fallback subtracted only rain, so
 * graupel and hail shafts rendered as cloud body.
 *
 * Run via:
 *   cd ui && npm run test:precipitating-species
 */

import {
  availablePrecipitatingSpecies,
  PRECIPITATING_SPECIES,
  sumFields,
} from "../../../../ui/src/lib/plot/precipitatingSpecies.ts";
import type { Npy3D } from "../../../../ui/src/lib/platform/npy.ts";

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

function check(label: string, fn: () => void): void {
  try {
    fn();
  } catch (e) {
    failed++;
    failures.push(`THREW ${label}: ${(e as Error).message}`);
  }
}

function field(shape: [number, number, number], value: number): Npy3D {
  const [nx, ny, nz] = shape;
  return { shape, data: new Float32Array(nx * ny * nz).fill(value) };
}

// ---------- Species list ----------

check("graupel and hail are removed alongside rain", () => {
  // total_condensate = qc+qr+qi+qs+qg+qh; leaving qg/qh in renders the hail
  // shaft as cloud, which in a supercell is where most of the mass is.
  for (const species of ["qr", "qg", "qh"]) {
    assertTrue(PRECIPITATING_SPECIES.includes(species), `${species} is precipitating`);
  }
});

check("suspended species are never removed", () => {
  // Match the engine's cloud_condensate = qc+qi+qs contract exactly. Snow can
  // sediment physically, but Nimbus intentionally includes it in frozen cloud.
  for (const species of ["qc", "qi", "qs", "qv"]) {
    assertTrue(!PRECIPITATING_SPECIES.includes(species), `${species} stays in the cloud body`);
  }
});

check("only the species a run published are selected", () => {
  assertTrue(
    JSON.stringify(availablePrecipitatingSpecies(["qc", "qr", "qg", "theta"])) === JSON.stringify(["qr", "qg"]),
    "picks the published subset in order",
  );
  assertTrue(availablePrecipitatingSpecies([]).length === 0, "none published");
  assertTrue(
    availablePrecipitatingSpecies(["qc", "qi"]).length === 0,
    "a warm-cloud run with no precip species selects nothing",
  );
});

// ---------- Summation ----------

check("the sum adds every present species", () => {
  const shape: [number, number, number] = [2, 3, 4];
  const total = sumFields([field(shape, 1), field(shape, 2), field(shape, 4)]);
  assertTrue(total != null, "sum exists");
  assertTrue(total!.data.every((v) => v === 7), "1 + 2 + 4 = 7 everywhere");
  assertTrue(JSON.stringify(total!.shape) === JSON.stringify(shape), "shape preserved");
});

check("absent species are skipped, not counted as zero-shaped", () => {
  const shape: [number, number, number] = [2, 2, 2];
  const total = sumFields([field(shape, 3), null, null]);
  assertTrue(total != null && total.data.every((v) => v === 3), "single present field passes through");
});

check("no species at all yields nothing to subtract", () => {
  assertTrue(sumFields([]) == null, "empty list");
  assertTrue(sumFields([null, null]) == null, "all absent");
});

check("mismatched shapes are refused rather than summed", () => {
  // Same element count, different split: summing these would add one cell to
  // another cell's neighbour and silently scramble the result.
  const a = field([2, 3, 4], 1);
  const b = field([4, 3, 2], 1);
  assertTrue(a.data.length === b.data.length, "the trap: equal element counts");
  assertTrue(sumFields([a, b]) == null, "refused on shape");
});

check("a differing element count is refused", () => {
  assertTrue(sumFields([field([2, 2, 2], 1), field([2, 2, 3], 1)]) == null, "refused");
});

check("the sum does not alias or mutate its inputs", () => {
  const shape: [number, number, number] = [2, 2, 2];
  const a = field(shape, 1);
  const b = field(shape, 2);
  const total = sumFields([a, b]);
  assertTrue(total!.data !== a.data && total!.data !== b.data, "new buffer");
  assertTrue(a.data.every((v) => v === 1), "first input unchanged");
  assertTrue(b.data.every((v) => v === 2), "second input unchanged");
});

check("identical immutable inputs reuse one derived sum across panes", () => {
  const shape: [number, number, number] = [2, 2, 2];
  const a = field(shape, 1);
  const b = field(shape, 2);
  const first = sumFields([a, b]);
  const second = sumFields([a, b]);
  const distinct = sumFields([a, field(shape, 2)]);
  assertTrue(first === second, "same source identities reuse the prepared aggregate");
  assertTrue(first !== distinct, "a distinct source field cannot alias the cached aggregate");
});

console.log(`\nprecipitating species tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  " + f);
  process.exit(1);
}
process.exit(0);
