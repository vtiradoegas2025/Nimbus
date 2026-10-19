import { spawnSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const [project, shard, mode] = process.argv.slice(2);
const match = /^(\d+)\/(\d+)$/.exec(shard ?? "");
if (!/^[a-z0-9-]+$/.test(project ?? "") || !match || (mode && mode !== "--list-only")) {
  throw new Error("Usage: node tests/ui/run_browser_partition.mjs <project> <index/count> [--list-only]");
}
const index = Number(match[1]);
const count = Number(match[2]);
if (index < 1 || index > count || count > 256) throw new Error("Invalid partition index/count");
const ui = fileURLToPath(new URL("../../ui/", import.meta.url));
const cli = path.join(ui, "node_modules/playwright/cli.js");
function discover(extra = []) {
  const result = spawnSync(process.execPath, [cli, "test", `--project=${project}`, "--list", "--reporter=list", ...extra], {
    cwd: ui, encoding: "utf8", maxBuffer: 16 * 1024 * 1024,
    env: { ...process.env, FORCE_COLOR: "0" },
  });
  if (result.error || result.status !== 0) {
    throw new Error(`Browser test discovery failed: ${result.error ?? result.stderr ?? result.stdout}`);
  }
  const lines = result.stdout.split(/\r?\n/).map(line => line.trim())
    .filter(line => line.startsWith(`[${project}] › `));
  if (lines.length === 0 || new Set(lines).size !== lines.length) {
    throw new Error("Browser discovery returned an empty or ambiguous test inventory");
  }
  return lines;
}
const all = discover();
// Spread adjacent expensive rendering cases across jobs instead of grouping a whole spec in one job.
const selected = all.filter((_, offset) => offset % count === index - 1);
if (selected.length === 0) throw new Error("Browser partition is empty");
const directory = path.join(ui, "build/browser-partitions");
mkdirSync(directory, { recursive: true });
const list = path.join(directory, `${project}-${index}-of-${count}.txt`);
writeFileSync(list, selected.join("\n") + "\n");
const resolved = discover(["--test-list", list]);
if (resolved.length !== selected.length || resolved.some((line, offset) => line !== selected[offset])) {
  throw new Error("Playwright resolved a different inventory from the requested partition");
}
writeFileSync(list.replace(/\.txt$/, ".json"), JSON.stringify({ project, index, count, total: all.length, selected }, null, 2) + "\n");
console.log(`${project} partition ${index}/${count}: ${selected.length} of ${all.length} tests; inventory verified`);
if (mode !== "--list-only") {
  const result = spawnSync(process.execPath, [cli, "test", `--project=${project}`, "--test-list", list, "--workers=1"], {
    cwd: ui, stdio: "inherit",
  });
  if (result.error) throw result.error;
  process.exitCode = result.status ?? 1;
}
