/*
 * Do the Plan-view and Cross-section plots render every field this run published?
 *
 * The 2-D plots accept "any field", so gating never rejects one — which means a field that
 * the heatmap cannot draw shows up as an empty panel rather than as an unavailable option.
 * This walks every published field and reports what SlicePlot would do with it, using the
 * app's own display-default logic rather than a restatement of it.
 *
 *   npm run audit2d -- ~/.nimbus/runs/<run>
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { basename, join } from "node:path";

import { parseNpy, fieldRange, type NpyN } from "../../../ui/src/lib/platform/npy.ts";
import { defaultScalarDisplay } from "../../../ui/src/lib/plot/scalarDisplayDefaults.ts";

/** SlicePlot draws a (nx, ny, nz) volume by slicing, and a (nx, ny) surface field directly. */
const RENDERABLE_RANKS = new Set([2, 3]);

interface FieldReport {
  name: string;
  rank: number;
  shape: string;
  min: number;
  max: number;
  nonFinite: number;
  units: string;
  colormap: string;
  rangeMode: string;
  verdict: "RENDERS" | "FLAT" | "UNSUPPORTED" | "UNREADABLE";
  flags: string[];
}

function latestStep(runDir: string): string {
  const steps = readdirSync(runDir)
    .filter((e) => e.startsWith("step_") && statSync(join(runDir, e)).isDirectory())
    .sort();
  if (steps.length === 0) throw new Error(`no step_* directories under ${runDir}`);
  return join(runDir, steps[steps.length - 1]);
}

function fieldUnitsOf(stepDir: string): Record<string, string> {
  try {
    const manifest = JSON.parse(readFileSync(join(stepDir, "manifest.json"), "utf8"));
    const units: Record<string, string> = {};
    for (const entry of manifest.fields ?? []) {
      if (entry?.field_id && entry.units) units[entry.field_id] = entry.units;
    }
    return units;
  } catch {
    return {};
  }
}

function load(stepDir: string, name: string): NpyN | null {
  try {
    const bytes = readFileSync(join(stepDir, `${name}.npy`));
    return parseNpy(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  } catch {
    return null;
  }
}

function nonFiniteFraction(field: NpyN): number {
  const d = field.data;
  const stride = Math.max(1, Math.floor(d.length / 200_000));
  let seen = 0;
  let bad = 0;
  for (let i = 0; i < d.length; i += stride) {
    seen++;
    if (!Number.isFinite(d[i])) bad++;
  }
  return seen === 0 ? 0 : bad / seen;
}

const runDir = process.argv[2];
if (!runDir) {
  console.error("usage: npm run audit2d -- <runDir>");
  process.exit(2);
}

const stepDir = latestStep(runDir);
const units = fieldUnitsOf(stepDir);
const names = readdirSync(stepDir)
  .filter((f) => f.endsWith(".npy"))
  .map((f) => f.slice(0, -".npy".length))
  .sort();

const reports: FieldReport[] = [];
for (const name of names) {
  const field = load(stepDir, name);
  if (!field) {
    reports.push({ name, rank: 0, shape: "-", min: NaN, max: NaN, nonFinite: 0,
      units: units[name] ?? "", colormap: "-", rangeMode: "-", verdict: "UNREADABLE", flags: [] });
    continue;
  }
  const rank = field.shape.length;
  const { min, max } = fieldRange(field);
  const nonFinite = nonFiniteFraction(field);
  const display = defaultScalarDisplay(name, units);
  const flags: string[] = [];

  // A signed quantity on a sequential ramp hides the sign change the field is read for.
  if (min < 0 && max > 0 && display.rangeMode === "auto") {
    flags.push("signed but not symmetric");
  }
  // An asymmetric range under a diverging ramp puts the neutral colour off zero.
  if (display.rangeMode === "symmetric" && !(min < 0 && max > 0)) {
    flags.push("symmetric range on a single-signed field");
  }
  if (nonFinite > 0) flags.push(`${(nonFinite * 100).toFixed(1)}% non-finite`);
  if (!units[name]) flags.push("no unit in manifest");

  reports.push({
    name, rank, shape: field.shape.join("x"), min, max, nonFinite,
    units: units[name] ?? "-",
    colormap: display.colormap,
    rangeMode: display.rangeMode,
    verdict: !RENDERABLE_RANKS.has(rank) ? "UNSUPPORTED" : !(max > min) ? "FLAT" : "RENDERS",
    flags,
  });
}

console.log(`run: ${basename(runDir)}   step: ${basename(stepDir)}   fields: ${reports.length}\n`);
console.log(
  `${"verdict".padEnd(12)}${"field".padEnd(30)}${"rank".padEnd(6)}${"units".padEnd(10)}`
  + `${"colormap".padEnd(10)}${"range".padEnd(11)}values`,
);
for (const r of reports) {
  const values = Number.isNaN(r.min) ? "-" : `${r.min.toPrecision(3)} .. ${r.max.toPrecision(3)}`;
  const flags = r.flags.length ? `   <- ${r.flags.join("; ")}` : "";
  console.log(
    `${r.verdict.padEnd(12)}${r.name.padEnd(30)}${String(r.rank).padEnd(6)}${r.units.padEnd(10)}`
    + `${r.colormap.padEnd(10)}${r.rangeMode.padEnd(11)}${values}${flags}`,
  );
}

const byVerdict = (v: FieldReport["verdict"]) => reports.filter((r) => r.verdict === v).length;
console.log(
  `\nRENDERS ${byVerdict("RENDERS")}   FLAT ${byVerdict("FLAT")}`
  + `   UNSUPPORTED ${byVerdict("UNSUPPORTED")}   UNREADABLE ${byVerdict("UNREADABLE")}`,
);
const flagged = reports.filter((r) => r.flags.length > 0);
if (flagged.length > 0) console.log(`${flagged.length} field(s) flagged above.`);
