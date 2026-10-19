/*
 * Completion prefix parsing.
 *
 * The parser's job is as much about REFUSING as offering. It runs on every
 * keystroke and hands a path to the interpreter to resolve by getattr, so the
 * cases that matter are the ones where the honest answer is "I cannot know
 * that without running your code" -- `make_plot().ax` -- and the ones where a
 * naive backwards scan finds a plausible-looking name that is not one, like
 * the `14` in `3.14`.
 *
 * Run:
 *   node --experimental-strip-types --import ../tests/ui/register-ts-resolve.mjs \
 *     ../tests/ui/unit/studio/test_python_completion.ts
 */

import {
  candidateIsVisible,
  parseCompletionPrefix,
} from "../../../../ui/src/lib/studio/pythonCompletion.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(`FAIL ${label}`);
}

function parsed(text: string) {
  const request = parseCompletionPrefix(text);
  if (!request) return null;
  return { path: request.path.join("."), partial: request.partial, backtrack: request.backtrack };
}

// ---------- 1. Bare identifiers ----------

check(parsed("ar")?.partial === "ar", "a bare partial is the thing being typed");
check(parsed("ar")?.path === "", "a bare partial resolves against the global namespace");
check(parsed("ar")?.backtrack === 2, "backtrack covers exactly the typed characters");
check(parsed("w = np")?.partial === "np", "only the token at the caret is considered");
check(parsed("for item in ra")?.partial === "ra", "a partial after keywords still parses");

// ---------- 2. Attribute chains ----------

check(parsed("np.")?.path === "np", "a trailing dot asks for that object's attributes");
check(parsed("np.")?.partial === "", "a trailing dot has nothing typed yet");
check(parsed("np.")?.backtrack === 0, "a trailing dot replaces nothing");
check(parsed("np.ar")?.path === "np", "a partial after a dot keeps the object");
check(parsed("np.ar")?.partial === "ar", "a partial after a dot is the typed part");
check(parsed("np.ar")?.backtrack === 2, "backtrack excludes the dot");
check(parsed("np.linalg.no")?.path === "np.linalg", "a multi-step chain is kept in order");
check(parsed("np.linalg.no")?.partial === "no", "the last segment of a chain is the partial");
check(parsed("ds.w.sel")?.path === "ds.w", "attribute chains through values parse");

// ---------- 3. Refusals ----------

/*
 * Each of these has a "reasonable-looking" wrong answer, which is why they are
 * enumerated rather than covered by one negative case.
 */
check(parsed("") === null, "empty input offers nothing");
check(parsed("   ") === null, "whitespace offers nothing");
check(parsed("w = ") === null, "open space after an operator offers nothing");
check(parsed("np.array(") === null, "an open call is not a completion site");

/*
 * The refusal that matters most: resolving this would mean CALLING make_plot to
 * ask what its result has on it. A popup appearing is not consent to run the
 * user's code.
 */
check(parsed("make_plot().ax") === null, "a call result is not resolved");
check(parsed("items[0].sh") === null, "a subscript result is not resolved");
check(parsed('"text".up') === null, "a literal's attributes are not resolved");
check(parsed(").fo") === null, "a dot after a closing paren is not resolved");

// A number is not a name, however much its tail looks like one.
check(parsed("3.14") === null, "a float literal is not an attribute access");
check(parsed("3.") === null, "a trailing dot on a number is not an attribute access");
check(parsed("x = 42") === null, "a bare integer is not a partial name");
check(parsed("a1") !== null, "an identifier containing digits still parses");
check(parsed("a1")?.partial === "a1", "an identifier containing digits parses whole");

// Underscores are identifier characters, at the start and inside.
check(parsed("_priv")?.partial === "_priv", "a leading underscore is an identifier");
check(parsed("np._")?.partial === "_", "a lone underscore after a dot is a partial");
check(parsed("my_var")?.partial === "my_var", "internal underscores are kept");

// ---------- 4. Hiding private names ----------

/*
 * dir() on a numpy array is over a hundred entries, most of them dunders. With
 * those in the list, the names a user actually wants are below the fold, which
 * makes the feature worse than not having it -- so they appear only once the
 * user has typed the underscore that asks for them.
 */
check(candidateIsVisible("mean", ""), "a public name is offered");
check(!candidateIsVisible("__array__", ""), "a dunder is hidden by default");
check(!candidateIsVisible("_private", ""), "a private name is hidden by default");
check(candidateIsVisible("_private", "_"), "typing an underscore reveals private names");
check(candidateIsVisible("__array__", "__"), "typing underscores reveals dunders");
check(candidateIsVisible("mean", "me"), "a matching public name stays visible");

console.log(`\npython completion tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length) {
  console.log("\nFailures:");
  for (const failure of failures) console.log(`  ${failure}`);
  process.exit(1);
}
