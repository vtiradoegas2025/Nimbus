import "./prepare.mjs";
import { readdirSync } from "node:fs";
import { join, relative } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const uiRoot = fileURLToPath(new URL("../../ui/", import.meta.url));
const testRoots = ["./unit/", "./release/"].map(path => fileURLToPath(new URL(path, import.meta.url)));

function collectTests(directory) {
  const tests = [];
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) tests.push(...collectTests(path));
    else if (/^test_.*\.ts$/.test(entry.name)) tests.push(path);
  }
  return tests;
}

const tests = testRoots.flatMap(collectTests).sort();
const failures = [];
if (tests.length === 0) throw new Error("No UI unit scripts discovered");

for (const testPath of tests) {
  const label = relative(uiRoot, testPath);
  process.stdout.write(`\n[unit] ${label}\n`);
  const result = spawnSync(
    process.execPath,
    [
      "--experimental-strip-types",
      "--import",
      "../tests/ui/register-ts-resolve.mjs",
      label,
    ],
    { cwd: uiRoot, encoding: "utf8", stdio: "inherit" },
  );
  if (result.error || result.status !== 0) {
    failures.push(`${label} (${result.error?.message ?? `exit ${result.status ?? "unknown"}`})`);
  }
}

if (failures.length > 0) {
  console.error(`\n${failures.length}/${tests.length} unit scripts failed:`);
  for (const failure of failures) console.error(`  - ${failure}`);
  process.exitCode = 1;
} else {
  console.log(`\nAll ${tests.length} unit scripts passed.`);
}
