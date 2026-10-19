/*
 * Which fields a script asks for, and what happens when they will not fit.
 *
 * The scan used to run over the raw document, so a name written in prose was
 * indistinguishable from a name used in code. The Python tab's own starter
 * header names eight fields it never touches; every Run therefore fetched all
 * of them, and on a 512x512x160 run that is 1.7 GiB pushed into a 4 GiB wasm
 * heap through a postMessage that copies. The interpreter did not raise — the
 * web content process was killed and the window reloaded.
 *
 * So the property under test is narrow and load-bearing: only code counts.
 * Everything else here defends the edges of that claim, because the stripping
 * is the kind of thing that silently over-reaches (dropping an f-string's real
 * expressions) or under-reaches (missing a triple-quoted block).
 *
 * Run:
 *   node --experimental-strip-types --import ../tests/ui/register-ts-resolve.mjs \
 *     ../tests/ui/unit/studio/test_python_field_refs.ts
 */

import {
  budgetExhausted,
  detectReferencedFields,
  fitsInBudget,
  MAX_AUTO_INJECT_BYTES,
  skippedFieldsNotice,
  stripPythonNonCode,
} from "../../../../ui/src/lib/studio/pythonFieldRefs.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(`FAIL ${label}`);
}

const FIELDS = ["u", "v", "w", "theta", "p", "rho", "qv", "qr", "vorticity_z", "theta_prime"];
const refs = (code: string) => detectReferencedFields(code, FIELDS);

// ---------- 1. Code counts ----------

check(
  refs("speed = np.hypot(u, v)").join() === "u,v",
  "a field used in an expression is requested",
);
check(refs("w = 1").length === 0, "a field assigned to is not loaded just to be overwritten");
check(refs("w = w + 1").join() === "w", "the right side of an assignment remains a field read");
check(refs("u, v = 1, 2").length === 0, "tuple assignment targets are not field reads");
check(refs("for w in range(3): pass").length === 0, "a loop target itself is not a field read");
check(refs("w += 1").join() === "w", "augmented assignment still reads the prior field value");
check(refs("").length === 0, "an empty document asks for nothing");
check(
  refs("import numpy as np\nx = 1").length === 0,
  "a document that names no field asks for nothing",
);

// ---------- 2. Prose does not ----------

check(
  refs("# u, v, w, theta, p, rho, qv, qr are available").length === 0,
  "the comment that cost 1.7 GiB a Run now costs nothing",
);
check(
  refs("x = u  # compare against theta and rho later").join() === "u",
  "a trailing comment does not smuggle in two more volumes",
);
check(
  refs('label = "vorticity_z over theta"').length === 0,
  "a field named inside a string is not a request for it",
);
check(
  refs('"""\nDocstring mentioning w and qv.\n"""\nx = 1').length === 0,
  "a triple-quoted block is prose too",
);
check(
  refs("x = u  # '\nrho = 2").join() === "u",
  "an apostrophe in a comment does not swallow the next assignment",
);
check(
  refs('await load_field("qr", frame=5)').length === 0,
  "load_field fetches its own argument, so naming it in a string injects nothing",
);

// ---------- 2b. An attribute is not a field ----------

check(
  detectReferencedFields("angle = np.pi * 2", ["pi", "u"]).length === 0,
  "np.pi is numpy's constant, not a field named pi — which some models do export",
);
check(
  detectReferencedFields("x = ds.theta.mean()", ["theta"]).length === 0,
  "an attribute chain does not request the field it happens to name",
);
check(
  refs("x = u.max()").join() === "u",
  "but the object the attribute hangs off is still a reference",
);
check(
  refs("x = 1.5 + w").join() === "w",
  "a decimal literal is not mistaken for an attribute and does not eat the next name",
);

// ---------- 3. f-strings are code where it matters ----------

check(
  refs('print(f"max is {w.max()}")').join() === "w",
  "an expression inside an f-string is real code",
);
check(
  refs('print(f"theta and rho are elsewhere: {u.max()}")').join() === "u",
  "the literal text around an f-string expression is still prose",
);
check(
  refs('print(f"{{theta}} is escaped, {v.min()} is not")').join() === "v",
  "a doubled brace is a literal brace, not an expression",
);

// ---------- 4. Stripping keeps tokens apart ----------

check(
  !/uv/.test(stripPythonNonCode('u"x"v')),
  "a removed string leaves a boundary rather than fusing its neighbours",
);
check(
  stripPythonNonCode('x = "a\\"b"\ny = w').includes("y = w"),
  "an escaped quote does not end the string early",
);
check(
  stripPythonNonCode("x = 1  # comment\ny = 2").includes("y = 2"),
  "a comment ends at its newline, not at the end of the file",
);

// ---------- 5. No field is loaded on the user's behalf ----------

for (const name of ["u", "v", "w", "theta", "p", "rho", "qv", "qr"]) {
  check(
    refs("x = 1").length === 0,
    `${name}: nothing is preloaded into a script that asked for nothing`,
  );
}

// ---------- 6. The budget ----------

const GIB = 1024 * 1024 * 1024;
check(fitsInBudget(0, MAX_AUTO_INJECT_BYTES), "a single field exactly at the budget fits");
check(!fitsInBudget(0, MAX_AUTO_INJECT_BYTES + 1), "one byte past the budget does not");
check(
  !fitsInBudget(MAX_AUTO_INJECT_BYTES, 1),
  "nothing more fits once the budget is spent",
);
check(MAX_AUTO_INJECT_BYTES < 4 * GIB, "the budget leaves room under the wasm32 ceiling");

// The measured case that started this: eleven 160 MiB fields on a 512x512x160
// run. Whatever the budget is, it must refuse that.
const BEAST_XL_FIELD = 512 * 512 * 160 * 4;
let admitted = 0;
let used = 0;
for (let i = 0; i < 11; i += 1) {
  if (!fitsInBudget(used, BEAST_XL_FIELD)) continue;
  used += BEAST_XL_FIELD;
  admitted += 1;
}
check(admitted < 11, "the eleven-field beast_xl case that killed the process is refused");
/*
 * Three 160 MiB fields is the case that actually failed, on WKWebView, with
 * `Unable to allocate 160. MiB` on the third. The budget has to land below it,
 * and above zero: a script naming one field must still get that field.
 */
check(admitted >= 1, "one field of that size still binds");
check(admitted < 3, "but not the three that WKWebView refused");

check(!budgetExhausted(0), "an untouched budget is not exhausted");
check(
  !budgetExhausted(MAX_AUTO_INJECT_BYTES - 1),
  "a budget with room left is not exhausted, so the next field is still read",
);
check(
  budgetExhausted(MAX_AUTO_INJECT_BYTES),
  "a spent budget stops the reads, not just the binds",
);

// ---------- 7. The refusal explains itself ----------

const notice = skippedFieldsNotice([
  { name: "theta", bytes: BEAST_XL_FIELD },
  { name: "qv", bytes: BEAST_XL_FIELD },
]);
check(notice.includes("theta") && notice.includes("qv"), "every skipped field is named");
check(/\d+ MiB/.test(notice), "with what it would have cost");
const unread = skippedFieldsNotice([{ name: "qr", bytes: null }]);
check(
  unread.includes("qr") && !unread.includes("MiB)"),
  "a field never read is still named, without inventing a size for it",
);
check(notice.includes("load_field"), "and the call that fetches one anyway");
check(
  skippedFieldsNotice([]) === "",
  "a run that skipped nothing says nothing",
);

console.log(`\npython field-ref tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length) {
  console.log("\nFailures:");
  for (const failure of failures) console.log(`  ${failure}`);
  process.exit(1);
}
