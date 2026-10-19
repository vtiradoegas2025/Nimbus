/*
 * Tests for `src/lib/plot/displayedRange.ts` — colouring a slice by what it actually contains.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types --import ../tests/ui/register-ts-resolve.mjs ../tests/ui/unit/plot/test_displayed_range.ts
 */

import { displayedRange, displayedRangeScope } from "../../../../ui/src/lib/plot/displayedRange.ts";

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

// ---------- The range itself ----------

check("the range covers the drawn cells", () => {
  const range = displayedRange([[1, 2, 3], [-4, 5, 0]]);
  assertTrue(range?.min === -4, "min is the smallest drawn value");
  assertTrue(range?.max === 5, "max is the largest drawn value");
});

check("a flat slice reports its constant, not an empty range", () => {
  /*
   * vorticity_z at the ground is identically zero on a beast_xl frame. Reporting null here
   * would let the caller fall back to the volume's range and paint the level flat, which is
   * the defect this module exists to remove; min == max is the honest answer.
   */
  const range = displayedRange([[0, 0], [0, 0]]);
  assertTrue(range?.min === 0 && range?.max === 0, "a constant slice ranges 0 .. 0");
});

check("non-finite cells are skipped rather than poisoning the range", () => {
  // A single NaN taken as data collapses the whole ramp.
  const range = displayedRange([[NaN, 2], [3, Infinity], [-1, -Infinity]]);
  assertTrue(range?.min === -1, "min ignores -Infinity and NaN");
  assertTrue(range?.max === 3, "max ignores +Infinity and NaN");
});

check("an entirely non-finite slice yields no range", () => {
  // Null tells the caller to keep the range it had, rather than snapping to 0 .. 0.
  assertTrue(displayedRange([[NaN, NaN], [NaN, NaN]]) === null, "all NaN -> null");
  assertTrue(displayedRange([]) === null, "no rows -> null");
  assertTrue(displayedRange([[]]) === null, "empty row -> null");
  assertTrue(displayedRange(null) === null, "null input -> null");
  assertTrue(displayedRange(undefined) === null, "undefined input -> null");
});

check("negative-only and positive-only slices keep their sign", () => {
  const negative = displayedRange([[-5, -1]]);
  assertTrue(negative?.min === -5 && negative?.max === -1, "negative-only stays negative");
  const positive = displayedRange([[1, 5]]);
  assertTrue(positive?.min === 1 && positive?.max === 5, "positive-only stays positive");
});

check("ragged rows are all visited", () => {
  const range = displayedRange([[1], [2, 3, 99], [0]]);
  assertTrue(range?.max === 99, "a longer row still contributes");
  assertTrue(range?.min === 0, "a shorter row still contributes");
});

// ---------- The latch scope ----------

check("the scope holds steady across frames", () => {
  // Time is not part of the identity: that is what keeps one physical value on one colour
  // through an animation instead of re-fitting the ramp every step.
  const first = displayedRangeScope("/run", "w", 2, 40);
  const second = displayedRangeScope("/run", "w", 2, 40);
  assertTrue(first === second, "same plane, same scope");
});

check("the scope changes with anything that changes the data", () => {
  const base = displayedRangeScope("/run", "w", 2, 40);
  assertTrue(displayedRangeScope("/run", "w", 2, 41) !== base, "a different level is a new scope");
  assertTrue(displayedRangeScope("/run", "w", 1, 40) !== base, "a different axis is a new scope");
  assertTrue(displayedRangeScope("/run", "theta", 2, 40) !== base, "a different field is a new scope");
  assertTrue(displayedRangeScope("/other", "w", 2, 40) !== base, "a different run is a new scope");
});

check("a surface field with no slice plane still gets a stable scope", () => {
  const first = displayedRangeScope("/run", "cape", null, null);
  assertTrue(first === displayedRangeScope("/run", "cape", null, null), "stable across frames");
  assertTrue(first !== displayedRangeScope("/run", "vil", null, null), "distinct per field");
});

check("scope fields cannot collide across the separator", () => {
  // Concatenating without a separator makes ("ab", "c") and ("a", "bc") the same scope, so
  // a range fitted to one field would be applied to another.
  assertTrue(
    displayedRangeScope("/run", "ab", 2, 1) !== displayedRangeScope("/run", "a", 2, 1),
    "field names do not bleed into the axis",
  );
  assertTrue(
    displayedRangeScope("/run", "w", 2, 11) !== displayedRangeScope("/run", "w", 21, 1),
    "axis and index stay distinct",
  );
});

// ---------- Summary ----------

for (const line of failures) console.error(line);
console.log(`\ndisplayed-range tests: ${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
