/*
 * Does each 3-D renderer actually draw geometry on a real run?
 *
 * The readiness report answers "are the fields present"; this answers the question that
 * follows it. An isosurface whose threshold sits outside the data range mounts cleanly,
 * reports no error, and renders an empty canvas — indistinguishable from a broken widget.
 * So each renderer is checked against the defaults it actually opens on.
 *
 *   npm run audit3d -- ~/.nimbus/runs/<run>
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { basename, join } from "node:path";

import { parseNpy, fieldRange, fieldPercentiles, type NpyN } from "../../../ui/src/lib/platform/npy.ts";
// Imported rather than transcribed: a hand-copied table drifted from the code once already,
// and produced a confident verdict about a threshold the app never uses.
import { physicalIsosurfaceThreshold } from "../../../ui/src/lib/plot/coherentStructures.ts";

/* Remaining defaults live inside component state, so they are mirrored here. */
/** With no physical default the widget opens at 60% of range, clamped into the slider's band. */
const ISOSURFACE_FALLBACK_FRACTION = 0.6;
const ISOSURFACE_SLIDER_MARGIN = 0.02;
/** Fields whose default surface is a signed pair; the physical value applies to |value|. */
const ISOSURFACE_SIGNED_PAIR = new Set([
  "vorticity_z", "vorticity_r", "vorticity_theta",
  "horizontal_vorticity_streamwise", "horizontal_vorticity_crosswise",
  "stretching_term", "tilting_term", "baroclinic_term",
]);
const SUPERCELL_UPDRAFT_THRESHOLD = 6.0; // m/s
const SUPERCELL_CLOUD_FLOOR_GKG = 0.1; // g/kg
const SUPERCELL_VORTEX_CORE_FRACTION = 0.45; // of max |value|

type Verdict = "DRAWS" | "EMPTY" | "DEGENERATE" | "MISSING";

interface Check {
  renderer: string;
  detail: string;
  verdict: Verdict;
  note: string;
}

function stepDirs(runDir: string): string[] {
  return readdirSync(runDir)
    .filter((e) => e.startsWith("step_") && statSync(join(runDir, e)).isDirectory())
    .sort()
    .map((e) => join(runDir, e));
}

function loadField(stepDir: string, name: string): NpyN | null {
  try {
    const bytes = readFileSync(join(stepDir, `${name}.npy`));
    // Copy out of the Buffer pool: a Buffer view can start mid-ArrayBuffer.
    return parseNpy(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  } catch {
    return null;
  }
}

/** Fraction of samples that are not finite; a field that is mostly NaN renders as holes. */
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

function describe(field: NpyN): string {
  const { min, max } = fieldRange(field);
  const nan = nonFiniteFraction(field);
  const shape = field.shape.join("x");
  const nanPart = nan > 0 ? `, ${(nan * 100).toFixed(1)}% non-finite` : "";
  return `[${shape}] ${min.toExponential(2)} .. ${max.toExponential(2)}${nanPart}`;
}

/** A scalar with no spread renders as a single flat value: no isosurface, no structure. */
function isDegenerate(field: NpyN): boolean {
  const { min, max } = fieldRange(field);
  return !(max > min);
}

function checkVolume(stepDir: string, name: string): Check {
  const field = loadField(stepDir, name);
  if (!field) return { renderer: "Volume", detail: name, verdict: "MISSING", note: "not exported" };
  if (field.shape.length !== 3) {
    return { renderer: "Volume", detail: name, verdict: "MISSING",
      note: `rank ${field.shape.length}; the volume renderer needs a 3-D array` };
  }
  if (isDegenerate(field)) {
    return { renderer: "Volume", detail: name, verdict: "DEGENERATE",
      note: `uniform field — ${describe(field)}` };
  }
  return { renderer: "Volume", detail: name, verdict: "DRAWS", note: describe(field) };
}

function checkIsosurface(stepDir: string, name: string): Check {
  const field = loadField(stepDir, name);
  if (!field) return { renderer: "Isosurface", detail: name, verdict: "MISSING", note: "not exported" };
  if (field.shape.length !== 3) {
    return { renderer: "Isosurface", detail: name, verdict: "MISSING",
      note: `rank ${field.shape.length}; marching cubes needs a 3-D array` };
  }
  if (isDegenerate(field)) {
    return { renderer: "Isosurface", detail: name, verdict: "DEGENERATE",
      note: `uniform field — ${describe(field)}` };
  }
  const { min, max } = fieldRange(field);
  const span = max - min;
  const physical = physicalIsosurfaceThreshold(name);
  const paired = ISOSURFACE_SIGNED_PAIR.has(name) && min < 0 && max > 0;

  let threshold: number;
  let source: string;
  if (paired && physical != null && physical > 0
      && physical <= Math.max(Math.abs(min), Math.abs(max))) {
    threshold = physical;
    source = "physical default (signed pair)";
  } else if (!paired && physical != null && physical >= min && physical <= max) {
    threshold = physical;
    source = "physical default";
  } else {
    const initial = min + ISOSURFACE_FALLBACK_FRACTION * span;
    const lo = min + ISOSURFACE_SLIDER_MARGIN * span;
    const hi = max - ISOSURFACE_SLIDER_MARGIN * span;
    threshold = Math.min(Math.max(initial, lo), hi);
    source = `${(ISOSURFACE_FALLBACK_FRACTION * 100).toFixed(0)}% of range`;
  }

  // A paired surface also draws the negative side, so either crossing is enough.
  const crosses = (threshold > min && threshold < max)
    || (paired && -threshold > min && -threshold < max);

  /*
   * Where the default lands in the distribution. A threshold up near p99 on a sparse field
   * draws only the densest core rather than the body of the structure — legible geometry,
   * but not the view the field is usually read for.
   */
  const [p50, p90, p99] = fieldPercentiles(field, [0.5, 0.9, 0.99]);
  let placement = "";
  if (crosses && Number.isFinite(p99)) {
    if (threshold >= p99) placement = "  <- above p99: densest core only";
    else if (threshold >= p90) placement = "  <- between p90 and p99";
    else if (threshold <= p50) placement = "  <- at or below the median";
  }

  return {
    renderer: "Isosurface",
    detail: name,
    verdict: crosses ? "DRAWS" : "EMPTY",
    note: `${source} ${threshold.toPrecision(3)} vs range ${min.toPrecision(3)} .. ${max.toPrecision(3)}`
      + placement,
  };
}

function checkStreamlines(stepDir: string): Check[] {
  const names = ["u", "v", "w"];
  const fields = names.map((n) => ({ name: n, field: loadField(stepDir, n) }));
  const missing = fields.filter((f) => !f.field).map((f) => f.name);
  if (missing.length > 0) {
    return [{ renderer: "Streamlines", detail: "u/v/w", verdict: "MISSING",
      note: `missing ${missing.join(", ")}` }];
  }
  const allDegenerate = fields.every((f) => isDegenerate(f.field!));
  return [{
    renderer: "Streamlines",
    detail: "u/v/w",
    verdict: allDegenerate ? "DEGENERATE" : "DRAWS",
    note: allDegenerate
      ? "every component is uniform — integration produces straight lines or nothing"
      : fields.map((f) => `${f.name} ${describe(f.field!)}`).join("  |  "),
  }];
}

function checkSupercell(stepDir: string): Check[] {
  const checks: Check[] = [];

  const cloudName = ["cloud_condensate", "total_condensate"].find((n) => loadField(stepDir, n));
  if (!cloudName) {
    checks.push({ renderer: "Supercell", detail: "cloud body", verdict: "MISSING",
      note: "neither cloud_condensate nor total_condensate is exported" });
  } else {
    const cloud = loadField(stepDir, cloudName)!;
    const { max } = fieldRange(cloud);
    // The control is in g/kg; the field is kg/kg.
    const floorKgKg = SUPERCELL_CLOUD_FLOOR_GKG * 1e-3;
    checks.push({
      renderer: "Supercell",
      detail: `cloud body (${cloudName})`,
      verdict: max > floorKgKg ? "DRAWS" : "EMPTY",
      note: `floor ${SUPERCELL_CLOUD_FLOOR_GKG} g/kg (${floorKgKg.toExponential(1)} kg/kg) vs max ${max.toExponential(2)}`,
    });
  }

  const w = loadField(stepDir, "w");
  if (!w) {
    checks.push({ renderer: "Supercell", detail: "updraft/downdraft", verdict: "MISSING", note: "w not exported" });
  } else {
    const { min, max } = fieldRange(w);
    const up = max > SUPERCELL_UPDRAFT_THRESHOLD;
    const down = min < -SUPERCELL_UPDRAFT_THRESHOLD;
    checks.push({
      renderer: "Supercell",
      detail: "updraft/downdraft (w)",
      verdict: up || down ? "DRAWS" : "EMPTY",
      note: `threshold +/-${SUPERCELL_UPDRAFT_THRESHOLD} m/s vs ${min.toPrecision(3)} .. ${max.toPrecision(3)}`
        + ` (updraft ${up ? "yes" : "no"}, downdraft ${down ? "yes" : "no"})`,
    });
  }

  const coreName = ["vorticity_z", "p_prime"].find((n) => loadField(stepDir, n));
  if (!coreName) {
    checks.push({ renderer: "Supercell", detail: "vortex cores", verdict: "MISSING",
      note: "neither vorticity_z nor p_prime is exported" });
  } else {
    const core = loadField(stepDir, coreName)!;
    const { min, max } = fieldRange(core);
    const threshold = SUPERCELL_VORTEX_CORE_FRACTION * Math.max(Math.abs(min), Math.abs(max));
    checks.push({
      renderer: "Supercell",
      detail: `vortex cores (${coreName})`,
      verdict: threshold > 0 ? "DRAWS" : "DEGENERATE",
      // Adaptive: a fraction of the frame's own extremum, so it tracks the data.
      note: `${(SUPERCELL_VORTEX_CORE_FRACTION * 100).toFixed(0)}% of max|v| = ${threshold.toExponential(2)}`
        + ` vs ${min.toPrecision(3)} .. ${max.toPrecision(3)}`,
    });
  }

  return checks;
}

const MARK: Record<Verdict, string> = {
  DRAWS: "DRAWS   ", EMPTY: "EMPTY   ", DEGENERATE: "FLAT    ", MISSING: "MISSING ",
};

function report(label: string, checks: Check[]): void {
  console.log(`\n--- ${label} ---`);
  for (const c of checks) {
    console.log(`  ${MARK[c.verdict]} ${c.renderer.padEnd(12)} ${c.detail.padEnd(34)} ${c.note}`);
  }
}

const runDir = process.argv[2];
if (!runDir) {
  console.error("usage: npm run audit3d -- <runDir>");
  process.exit(2);
}

const steps = stepDirs(runDir);
if (steps.length === 0) {
  console.error(`no step_* directories under ${runDir}`);
  process.exit(2);
}

console.log(`run: ${basename(runDir)}  (${steps.length} steps)`);

/*
 * First and last step both matter. At t=0 a forcing-free field is legitimately flat, so an
 * empty isosurface there is correct rather than a defect — and reporting only the last step
 * would hide that the widget opens on an empty scene when a user first loads a run.
 */
const volumeCandidates = ["reflectivity_dbz", "vorticity_magnitude", "w", "theta_prime"];
const isoCandidates = ["w", "vorticity_magnitude", "vorticity_z", "cloud_condensate"];

for (const [label, stepDir] of [
  [`first step (${basename(steps[0])})`, steps[0]],
  [`last step (${basename(steps[steps.length - 1])})`, steps[steps.length - 1]],
] as const) {
  const checks: Check[] = [
    ...volumeCandidates.map((n) => checkVolume(stepDir, n)),
    ...isoCandidates.map((n) => checkIsosurface(stepDir, n)),
    ...checkStreamlines(stepDir),
    ...checkSupercell(stepDir),
  ];
  report(label, checks);
}

console.log("\nDRAWS = geometry appears at the widget's default settings."
  + "\nEMPTY = mounts cleanly but shows nothing; the default threshold misses the data."
  + "\nFLAT  = field has no spread. MISSING = not exported, or wrong rank.\n");
