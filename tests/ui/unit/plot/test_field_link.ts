/*
 * Tests for `src/lib/plot/fieldLink.ts` — one pane following another's field.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types --import ../tests/ui/register-ts-resolve.mjs ../tests/ui/unit/plot/test_field_link.ts
 */

import {
  resolveFieldLink,
  describeDeclinedFieldLink,
  SINGLE_LEVEL_LINK_REASON,
} from "../../../../ui/src/lib/plot/fieldLink.ts";

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

const anything = () => true;
const nothing = () => false;
const volumetric = (f: string) => !["cape", "cin", "vil", "composite_reflectivity"].includes(f);

// ---------- Following ----------

check("a pane follows a field it can draw", () => {
  const decision = resolveFieldLink("theta", "w", anything);
  assertTrue(decision.adopt === "theta", "adopts the linked field");
  assertTrue(decision.declined === null, "nothing refused");
});

check("a pane already on the field does nothing", () => {
  /*
   * The no-op matters: adopting sets state, and a pane that re-adopted its own field every
   * render would loop against the pane it is linked to.
   */
  const decision = resolveFieldLink("w", "w", anything);
  assertTrue(decision.adopt === null, "no switch");
  assertTrue(decision.declined === null, "nothing refused");
});

check("no link means no change", () => {
  for (const linked of [null, undefined, ""]) {
    const decision = resolveFieldLink(linked, "w", anything);
    assertTrue(decision.adopt === null, `${String(linked)} leaves the pane alone`);
    assertTrue(decision.declined === null, `${String(linked)} refuses nothing`);
  }
});

// ---------- Refusing ----------

check("a renderer that cannot draw the field keeps its own", () => {
  /*
   * The alternative — following anyway — leaves a volume renderer pointed at a field with no
   * volume, which reports a shape error and shows an empty canvas while the toggle still reads
   * as linked.
   */
  const decision = resolveFieldLink("cape", "w", volumetric);
  assertTrue(decision.adopt === null, "does not switch");
  assertTrue(decision.declined === "cape", "reports what it refused");
});

check("a pane that can draw nothing refuses rather than blanking", () => {
  const decision = resolveFieldLink("theta", "w", nothing);
  assertTrue(decision.adopt === null, "stays on its field");
  assertTrue(decision.declined === "theta", "and says which field it refused");
});

check("refusing does not depend on having a current field", () => {
  const decision = resolveFieldLink("cape", null, volumetric);
  assertTrue(decision.adopt === null, "no switch");
  assertTrue(decision.declined === "cape", "still explains itself");
});

check("a pane with no field adopts the first one it can draw", () => {
  const decision = resolveFieldLink("theta", null, anything);
  assertTrue(decision.adopt === "theta", "adopts into an empty pane");
});

// ---------- The decision never both adopts and refuses ----------

check("adopt and declined are mutually exclusive", () => {
  const cases: Array<[string | null, string | null, (f: string) => boolean]> = [
    ["theta", "w", anything],
    ["cape", "w", volumetric],
    ["w", "w", anything],
    [null, "w", anything],
    ["theta", null, nothing],
  ];
  for (const [linked, current, canShow] of cases) {
    const d = resolveFieldLink(linked, current, canShow);
    assertTrue(
      !(d.adopt != null && d.declined != null),
      `("${linked}", "${current}") does not both adopt and refuse`,
    );
  }
});

check("the predicate is consulted only when a switch is possible", () => {
  // Guards against asking an expensive predicate on every render for a no-op.
  let calls = 0;
  const counting = (f: string) => { calls++; return f.length > 0; };
  resolveFieldLink("w", "w", counting);
  assertTrue(calls === 0, "same field: predicate not consulted");
  resolveFieldLink(null, "w", counting);
  assertTrue(calls === 0, "no link: predicate not consulted");
  resolveFieldLink("theta", "w", counting);
  assertTrue(calls === 1, "a real candidate consults it once");
});

// ---------- The note ----------

check("the refusal names the field and the reason", () => {
  const note = describeDeclinedFieldLink("cape", SINGLE_LEVEL_LINK_REASON);
  assertTrue(note.includes("cape"), "names the field");
  assertTrue(note.includes("vertical dimension"), "gives the reason");
});

// ---------- Summary ----------

for (const line of failures) console.error(line);
console.log(`\nfield-link tests: ${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
