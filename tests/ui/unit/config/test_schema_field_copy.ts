/*
 * Tests for the copy the config form shows on fields the user cannot edit.
 *
 * A padlock or a grayed-out control is only usable if it says WHY. Two faults
 * motivated these checks:
 *   1. SchemaForm hardcoded one padlock tooltip -- "Derived from the Scheme
 *      selection -- pick the Scheme to change coordinate, staggering, and
 *      axisymmetric together" -- for every locked field. The only field that
 *      ever renders it is grid.coordinate_system, and the sentence is wrong
 *      there twice over: dynamics.scheme is hidden (visibleWhen -> false) so
 *      the named control does not exist, and the coupling runs the other way
 *      (reconcileGeometry derives the scheme FROM the staggering).
 *   2. extensions.validation.guard_mode is disabled under Qualified execution
 *      with no on-screen reason at all -- the explanation lived in the
 *      `description`, which is collapsed behind the info button by default.
 *
 * The invariants below are structural rather than string-matching, so a new
 * locked or conditionally-disabled field cannot ship without stating a reason.
 *
 * Run via:
 *   cd ui && npm run test:schema-copy
 */

import {
  SCHEMA_SECTIONS,
  defaultsForSchema,
  applyFieldChange,
  qualifyKey,
  readPath,
  type ConfigObject,
  type FieldSchema,
  type SectionSchema,
} from "../../../../ui/src/lib/config/configSchema.ts";

let passed = 0;
let failed = 0;
const failures: string[] = [];

function check(label: string, condition: boolean): void {
  if (condition) {
    passed++;
  } else {
    failed++;
    failures.push(label);
  }
}

/** Fields the form can actually render, paired with their qualified path. */
/** The sub-tree visibleWhen receives, matching SchemaForm's own resolution. */
function sectionSubtree(section: SectionSchema, root: ConfigObject): ConfigObject {
  if (!section.path) return root;
  return (readPath(root, section.path) ?? {}) as ConfigObject;
}

function visibleFields(
  section: SectionSchema,
  root: ConfigObject,
): Array<{ field: FieldSchema; path: string }> {
  const sectionRoot = sectionSubtree(section, root);
  return section.fields
    .filter((field) => !field.visibleWhen || field.visibleWhen(sectionRoot))
    .map((field) => ({ field, path: qualifyKey(section, field) }));
}

const defaults = defaultsForSchema();

/* ---------- Every renderable locked field explains its lock ---------- */

let lockedCount = 0;
for (const section of SCHEMA_SECTIONS) {
  for (const { field, path } of visibleFields(section, defaults)) {
    if (field.locked !== true) continue;
    lockedCount += 1;
    check(
      `${path} states why it is locked`,
      typeof field.lockedReason === "string" && field.lockedReason.trim().length > 0,
    );
  }
}
check("at least one locked field is renderable (the check is not vacuous)", lockedCount > 0);

/* ---------- No lock reason may point at a control the form hides ----------
 * A reason naming a hidden field sends the user hunting for a control that is
 * not on screen. dynamics.scheme and dynamics.axisymmetric are both hidden. */

const hiddenFieldLabels = SCHEMA_SECTIONS.flatMap((section) => {
  const sectionRoot = sectionSubtree(section, defaults);
  return section.fields
    .filter((field) => field.visibleWhen != null && !field.visibleWhen(sectionRoot))
    .map((field) => field.key);
});
check("dynamics.scheme is hidden, so no reason may name it", hiddenFieldLabels.includes("scheme"));

for (const section of SCHEMA_SECTIONS) {
  for (const { field, path } of visibleFields(section, defaults)) {
    const reason = `${field.lockedReason ?? ""} ${field.disabledReason ?? ""}`.toLowerCase();
    if (!reason.trim()) continue;
    check(
      `${path} does not send the user to the hidden Scheme control`,
      !/\bscheme selection\b|\bpick the scheme\b/.test(reason),
    );
  }
}

/* ---------- Every conditionally-disabled field explains the condition ---------- */

let disabledCount = 0;
for (const section of SCHEMA_SECTIONS) {
  for (const { field, path } of visibleFields(section, defaults)) {
    if (!field.disabledWhen) continue;
    disabledCount += 1;
    check(
      `${path} states why it is disabled`,
      typeof field.disabledReason === "string" && field.disabledReason.trim().length > 0,
    );
    check(
      `${path} names the setting that owns it`,
      /execution|run mode|qualified|staggering|backend|coordinate/i.test(field.disabledReason ?? ""),
    );
  }
}
check("at least one conditionally-disabled field exists", disabledCount > 0);

/* ---------- guard_mode: the predicate, the rewrite, and the reason agree ---------- */

const guardField = SCHEMA_SECTIONS.flatMap((s) =>
  s.fields.filter((f) => f.key === "guard_mode"),
)[0];

const cudaField = SCHEMA_SECTIONS.flatMap((section) => section.fields).find((field) => field.key === "cuda_assist");
check("CUDA assist is explicit and defaults off", cudaField?.default === false);
const vulkanConfig = applyFieldChange(defaults, "extensions.compute.backend", "vulkan");
const cudaConfig = applyFieldChange(vulkanConfig, "extensions.compute.cuda_assist", true);
check("CUDA assist is visible for Vulkan", cudaField?.visibleWhen?.(readPath(cudaConfig, "extensions.compute") as ConfigObject) === true);
for (const backend of ["cpu", "metal"]) {
  const switched = applyFieldChange(cudaConfig, "extensions.compute.backend", backend);
  check(`switching to ${backend} clears CUDA assist`, readPath(switched, "extensions.compute.cuda_assist") === false);
  check(`CUDA assist is hidden for ${backend}`, cudaField?.visibleWhen?.(readPath(switched, "extensions.compute") as ConfigObject) === false);
}
check("guard_mode is in the schema", guardField != null);

if (guardField?.disabledWhen) {
  const qualified = applyFieldChange(defaults, "extensions.execution.mode", "qualified");
  check("guard_mode disables under Qualified execution", guardField.disabledWhen(qualified));
  check("guard_mode stays editable otherwise", !guardField.disabledWhen(defaults));
  // The control is not merely grayed out: its value is rewritten underneath the
  // user, so the reason has to account for the new value as well as the lock.
  check(
    "Qualified execution pins guard_mode to error",
    readPath(qualified, "extensions.validation.guard_mode") === "error",
  );
  check(
    "the disabled reason mentions the pinned value",
    /error/i.test(guardField.disabledReason ?? ""),
  );
}

/* ---------- The GPU hint may not promise a fallback Qualified mode removes ----------
 * configPreflight sets allowBackendFallback = (mode !== "qualified"), so an
 * unconditional "auto-falls-back to CPU" is false in exactly the mode whose own
 * description advertises failing closed. */

const backendField = SCHEMA_SECTIONS.flatMap((s) =>
  s.fields.filter((f) => f.key === "backend"),
)[0];
check("compute backend is in the schema", backendField != null);
if (backendField?.hint && /falls?[- ]back/i.test(backendField.hint)) {
  check(
    "the backend fallback hint carries the Qualified-mode exception",
    /qualified/i.test(backendField.hint),
  );
}

console.log(`\nschema-field-copy tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  " + f);
  process.exit(1);
}
process.exit(0);
