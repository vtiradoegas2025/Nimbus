/* Regression tests for incremental run-log classification and counters. */

import {
  appendLogBatch,
  classifyLine,
  type LogLine,
  type RunLogSnapshot,
} from "../../../../ui/src/lib/run/runLog.ts";

let passed = 0;
let failed = 0;
const failures: string[] = [];

function check(label: string, condition: boolean): void {
  if (condition) passed++;
  else {
    failed++;
    failures.push(label);
  }
}

function line(id: number, text: string, stream: "stdout" | "stderr" = "stdout"): LogLine {
  const input = { stream, line: text };
  return { id, ...input, ts: id, kind: classifyLine(input) };
}

const empty: RunLogSnapshot = { lines: [], warnCount: 0, errorCount: 0 };
const first = appendLogBatch(empty, [
  line(1, "[INFO] ready"),
  line(2, "[WARN] high CFL"),
  line(3, "worker failed", "stderr"),
]);
check("batch retains every line below the bound", first.lines.length === 3);
check("warning count is updated while appending", first.warnCount === 1);
check("stderr contributes to the error count", first.errorCount === 1);
check("classification is stored on the retained row", first.lines[1].kind === "warn");

const second = appendLogBatch(first, [
  line(4, "[FATAL] unstable"),
  line(5, "[INFO] stopped"),
], 3);
check("ring buffer stays within its requested bound", second.lines.length === 3);
check("old retained warning is removed from the counter", second.warnCount === 0);
check("retained stderr and fatal lines are both counted", second.errorCount === 2);
check("the newest ids survive eviction", second.lines.map(({ id }) => id).join(",") === "3,4,5");

const unchanged = appendLogBatch(second, []);
check("an empty batch preserves snapshot identity", unchanged === second);

console.log(`\nrun-log tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const failure of failures) console.log(`  ${failure}`);
  process.exit(1);
}

