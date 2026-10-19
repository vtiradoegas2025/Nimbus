/*
 * Prints the Plot readiness panel's verdict for a real run directory, from the
 * command line, using the same registry predicates the UI uses.
 *
 * The readiness panel is the app's own answer to "what can this run show me?",
 * so it is the natural work list for closing coverage gaps — but reading it
 * requires launching the app and expanding a <details>. This renders the same
 * computation against a run on disk, so the list can be diffed as fields are
 * added and as gating is corrected.
 *
 *   npm run readiness -- ~/.nimbus/runs/<run>
 */
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { basename, join } from "node:path";

import {
  CATEGORY_META,
  TRAJECTORY_CAPABILITY,
  isPlotAvailable,
  isPresetAvailable,
  metaFor,
  missingRequirementClauses,
  presetRequirement,
  type FieldReq,
} from "../../../ui/src/lib/plot/plotRegistry.ts";
import { PLOT_PRESETS } from "../../../ui/src/lib/plot/plotPresets.ts";
import {
  describeMissingClause,
  groupRemedyAdvice,
  type MissingClauseAdvice,
} from "../../../ui/src/lib/plot/fieldProvenance.ts";
import { augmentStepWithDerivedFields } from "../../../ui/src/lib/derived/derivedFieldRegistry.ts";
import type { StepInfo } from "../../../ui/src/lib/run/runOutputs.ts";

interface ManifestField {
  field_id: string;
  units?: string;
}

/**
 * The field set the app would gate on, reproduced from a run on disk.
 *
 * Two details matter and both were got wrong by a first pass that just listed
 * `.npy` files: a step manifest enumerates the engine's whole field *catalog*
 * rather than what this run wrote, so availability must come from the files;
 * and Studio's builtin derived fields are appended to every step by
 * `augmentStepWithDerivedFields`, so a run that publishes theta/p/qv already
 * offers dewpoint, relative humidity, and theta_e without exporting them.
 */
function studioAvailableFields(runDir: string): Set<string> {
  const written = new Set<string>();
  let fieldUnits: Record<string, string> = {};
  for (const entry of readdirSync(runDir)) {
    if (!entry.startsWith("step_")) continue;
    const stepDir = join(runDir, entry);
    if (!statSync(stepDir).isDirectory()) continue;
    for (const file of readdirSync(stepDir)) {
      if (file.endsWith(".npy")) written.add(file.slice(0, -".npy".length));
    }
    const manifestPath = join(stepDir, "manifest.json");
    if (Object.keys(fieldUnits).length === 0 && existsSync(manifestPath)) {
      const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as { fields?: ManifestField[] };
      fieldUnits = Object.fromEntries(
        (manifest.fields ?? [])
          .filter((field) => written.has(field.field_id) && field.units)
          .map((field) => [field.field_id, field.units as string]),
      );
    }
  }
  const step = augmentStepWithDerivedFields({
    index: 0,
    fields: [...written],
    fieldUnits,
    publicationState: "complete",
    publicationGeneration: `readiness:${runDir}`,
  } as StepInfo);
  const available = new Set(step.fields);
  // The same artifact `run_has_trajectories` probes in src-tauri: one run-level
  // file at the outdir root, not a per-step directory.
  if (existsSync(join(runDir, "trajectories.npy"))) available.add(TRAJECTORY_CAPABILITY);
  return available;
}

function describeGap(req: FieldReq, available: ReadonlySet<string>): string {
  const missing = missingRequirementClauses(req, available);
  if (missing.length === 0) return "ready";
  if (missing.some((clause) => clause.length === 0)) return "needs any readable field";
  return missing.map((clause) => clause.join(" or ")).join(" + ");
}

const runDir = process.argv[2];
if (!runDir) {
  console.error("usage: readiness_report.ts <runDir>");
  process.exit(2);
}

const available = studioAvailableFields(runDir);
console.log(`run: ${basename(runDir)}`);
console.log(`published fields (${available.size}): ${[...available].sort().join(", ")}\n`);

let ready = 0;
let total = 0;
const blocked: Array<{ category: string; label: string; kind: string; gap: string }> = [];
const requirements: FieldReq[] = [];

for (const category of CATEGORY_META) {
  const renderers = category.plotTypes.map(metaFor);
  const presets = PLOT_PRESETS.filter((p) => p.category === category.value);
  for (const renderer of renderers) {
    total += 1;
    requirements.push(renderer.requires);
    if (isPlotAvailable(renderer.value, available)) ready += 1;
    else {
      blocked.push({
        category: category.label,
        label: renderer.label,
        kind: "renderer",
        gap: describeGap(renderer.requires, available),
      });
    }
  }
  for (const preset of presets) {
    total += 1;
    requirements.push(presetRequirement(preset));
    if (isPresetAvailable(preset, available)) ready += 1;
    else {
      blocked.push({
        category: category.label,
        label: preset.label,
        kind: "preset",
        gap: describeGap(presetRequirement(preset), available),
      });
    }
  }
}

console.log(`${ready} of ${total} plot options ready; ${blocked.length} blocked\n`);
for (const item of blocked) {
  console.log(`  [${item.category}] ${item.label} (${item.kind}) — needs ${item.gap}`);
}

/*
 * The remedy list, exactly as the readiness panel renders it: one entry per
 * missing product rather than per blocked view, actionable products first.
 * Printing it here is how the advice gets reviewed against a real run without
 * launching the app and expanding a <details>.
 */
const remedies = new Map<string, { advice: MissingClauseAdvice; views: number }>();
for (const req of requirements) {
  for (const clause of missingRequirementClauses(req, available)) {
    const key = clause.join("|");
    const existing = remedies.get(key);
    if (existing) existing.views += 1;
    else remedies.set(key, { advice: describeMissingClause(clause, "nimbus_run"), views: 1 });
  }
}
const ordered = [...remedies.values()].sort((left, right) => right.views - left.views);

console.log(`\nwhat each missing product needs (${ordered.length} products):`);
for (const { title, action, entries } of groupRemedyAdvice(ordered, ({ advice }) => advice, "nimbus_run")) {
  console.log(`\n  ${title} (${entries.length})`);
  console.log(`  ${action}`);
  for (const { advice, views } of entries) {
    const term = advice.fields.join(" or ") || "any readable field";
    console.log(`      ${term.padEnd(26)} ${advice.detail}${advice.detail ? " · " : ""}blocks ${views}`);
  }
}
