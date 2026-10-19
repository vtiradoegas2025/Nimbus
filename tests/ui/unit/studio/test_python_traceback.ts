/*
 * Traceback linking.
 *
 * Two properties, both of which fail silently if broken:
 *
 *   1. The text is preserved exactly. A traceback is evidence; a renderer that
 *      dropped or reordered a frame while making others clickable would be
 *      lying about what happened, and nothing on screen would say so.
 *
 *   2. Only the user's frames link. Pyodide's own machinery appears above every
 *      user frame with real stdlib paths, and a link to `_base.py` line 597
 *      would put the caret on line 597 of the user's script -- a real line,
 *      unrelated to the error, with nothing to indicate the mismatch.
 *
 * Run:
 *   node --experimental-strip-types --import ../tests/ui/register-ts-resolve.mjs \
 *     ../tests/ui/unit/studio/test_python_traceback.ts
 */

import { offsetForLine, splitTraceback } from "../../../../ui/src/lib/studio/pythonTraceback.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(`FAIL ${label}`);
}

/* A real Pyodide traceback, as captured from the browser suite. */
const REAL_TRACEBACK = [
  "Traceback (most recent call last):",
  '  File "/lib/python312.zip/_pyodide/_base.py", line 597, in eval_code_async',
  "    await CodeRunner(",
  '  File "/lib/python312.zip/_pyodide/_base.py", line 411, in run_async',
  "    coroutine = eval(self.code, globals, locals)",
  '  File "<exec>", line 6, in <module>',
  "ValueError: marker-xyz",
  "",
].join("\n");

// ---------- 1. The text survives ----------

const segments = splitTraceback(REAL_TRACEBACK);
check(
  segments.map((segment) => segment.text).join("") === REAL_TRACEBACK,
  "concatenating the segments reproduces the traceback exactly",
);
check(splitTraceback("").length === 0, "empty stderr yields no segments");
check(
  splitTraceback("plain text").map((s) => s.text).join("") === "plain text",
  "text with no frames passes through whole",
);
check(
  splitTraceback("plain text").every((segment) => segment.line === null),
  "text with no frames produces no links",
);

// ---------- 2. Only the user's frames link ----------

const linked = segments.filter((segment) => segment.line !== null);
check(linked.length === 1, "exactly one frame in a real traceback is the user's");
check(linked[0].line === 6, "the linked frame carries its line number");
check(
  linked[0].text === 'File "<exec>", line 6',
  "the link covers the frame header and nothing more",
);
/*
 * The stdlib frames also read `line NNN`. Matching those is the mistake this
 * guards: both stdlib lines here (597, 411) exist in plenty of user scripts.
 */
check(
  !linked.some((segment) => segment.line === 597 || segment.line === 411),
  "stdlib frames are not linked",
);
check(
  segments.some((segment) => segment.line === null && segment.text.includes("_base.py")),
  "stdlib frames are still shown, just as plain text",
);

// Several user frames, as a recursive call produces.
const recursive = [
  '  File "<exec>", line 3, in <module>',
  '  File "<exec>", line 12, in helper',
  "RecursionError: maximum recursion depth exceeded",
].join("\n");
const recursiveLinks = splitTraceback(recursive).filter((s) => s.line !== null);
check(recursiveLinks.length === 2, "every user frame links");
check(
  recursiveLinks[0].line === 3 && recursiveLinks[1].line === 12,
  "user frames keep their order and numbers",
);
check(
  splitTraceback(recursive).map((s) => s.text).join("") === recursive,
  "multi-frame text is preserved exactly",
);

/*
 * Repeated calls agree. This is structurally guaranteed now that the splitter
 * uses matchAll rather than driving the module-level /g regex with exec, and it
 * is kept as a regression guard on that choice -- an exec loop would reintroduce
 * shared lastIndex state, which goes wrong only on the second call.
 *
 * Worth saying plainly: mutating the reset out of the previous exec-based
 * implementation did NOT fail this, because exec resets lastIndex itself once
 * it returns null and the loop always ran to exhaustion. The line it was
 * "covering" was dead.
 */
check(
  splitTraceback(REAL_TRACEBACK).filter((s) => s.line !== null).length === 1,
  "a repeated call returns the same links",
);

// ---------- 3. Line to offset ----------

const source = "alpha\nbeta\ngamma\n";
check(offsetForLine(source, 1) === 0, "line 1 starts at offset 0");
check(offsetForLine(source, 2) === 6, "line 2 starts after the first newline");
check(offsetForLine(source, 3) === 11, "line 3 starts after the second newline");
check(source.slice(offsetForLine(source, 2), 10) === "beta", "the offset lands on the line");
check(offsetForLine(source, 0) === 0, "a nonsensical line clamps to the start");
check(offsetForLine("", 5) === 0, "an empty document clamps to 0");

/*
 * A traceback can outlive the edit that invalidated it, so it will name lines
 * that no longer exist. Landing at the end is a poor answer; throwing from a
 * click handler is not an answer at all.
 */
check(
  offsetForLine(source, 99) === source.length,
  "a line past the end clamps to the end rather than throwing",
);
check(
  offsetForLine("no trailing newline", 2) === "no trailing newline".length,
  "a single-line document clamps for line 2",
);

console.log(`\npython traceback tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length) {
  console.log("\nFailures:");
  for (const failure of failures) console.log(`  ${failure}`);
  process.exit(1);
}
