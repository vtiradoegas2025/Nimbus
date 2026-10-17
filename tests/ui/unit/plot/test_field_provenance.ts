/**
 * Tests for `src/lib/plot/fieldProvenance.ts`.
 *
 * The point of this file is the first section: `NIMBUS_OUTPUT_FIELDS` is a
 * hand-carried copy of a C++ header in a sibling directory, and a stale copy
 * would make the readiness panel confidently wrong in exactly the way this
 * module exists to fix. So the header is re-parsed here and compared both
 * directions — a field added to the engine and not to the mirror fails, and so
 * does a field invented in the mirror.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types \
 *     --import ../tests/ui/register-ts-resolve.mjs \
 *     ../tests/ui/unit/plot/test_field_provenance.ts
 */

import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { SCHEMA_SECTIONS } from "../../../../ui/src/lib/config/configSchema.ts";
import {
  NIMBUS_OUTPUT_FIELDS,
  STUDIO_CATALOG_PRESET,
  STUDIO_CATALOG_PRESET_FIELDS,
  WIDEST_OUTPUT_PRESET,
  describeMissingClause,
  describeMissingFieldRemedy,
  describeRemedyGroup,
  fieldIsForeignToSource,
  fieldOrigins,
  groupRemedyAdvice,
  isVolumetricField,
  nimbusFieldKind,
  volumetricFields,
  presetPublishing,
  presetPublishingAll,
  summarizeMissingFieldReason,
  type NimbusFieldKind,
} from "../../../../ui/src/lib/plot/fieldProvenance.ts";
import {
  PLOT_TYPE_META,
  TRAJECTORY_CAPABILITY,
  presetRequirement,
  type FieldReq,
} from "../../../../ui/src/lib/plot/plotRegistry.ts";
import { PLOT_PRESETS } from "../../../../ui/src/lib/plot/plotPresets.ts";
import {
  compressionDisclosures,
  compressionDisclosuresAcrossSteps,
} from "../../../../ui/src/lib/plot/compressionProvenance.ts";
import type { StepInfo } from "../../../../ui/src/lib/run/runOutputs.ts";

// ---------- Tiny assertion API (matches test_radar_floor.ts) ----------

let failed = 0;
let passed = 0;
const failures: string[] = [];

function assertTrue(cond: boolean, label: string): void {
  if (cond) passed++;
  else {
    failed++;
    failures.push(`FAIL ${label}`);
  }
}

function check(label: string, fn: () => void): void {
  try {
    fn();
  } catch (e) {
    failed++;
    failures.push(`THREW ${label}: ${(e as Error).message}`);
  }
}

// ---------- Helpers ----------

const HERE = dirname(fileURLToPath(import.meta.url));
const REGISTRY_HEADER = resolve(HERE, "../../../../include/core/output/field_registry.hpp");
const OUTPUT_CONFIG_SOURCE = resolve(HERE, "../../../../src/core/output/output_config.cpp");

/** The C++ enumerator names, mapped onto this module's shape classes. Two
 *  enumerators collapse onto "surface" because both write one (nr, nth) plane;
 *  they differ only in how the engine assembles them. */
const KIND_BY_ENUMERATOR: Readonly<Record<string, NimbusFieldKind>> = {
  Core3D: "core_volume",
  DerivedVolume: "derived_volume",
  SurfaceTranche: "surface",
  DirectSurface2D: "surface",
  ColumnTranche: "column",
  RadarSynthetic: "radar_synthetic",
};

/** Parse the descriptor table out of field_registry.hpp. Each row looks like
 *  `{"cape", "cape", OutputFieldKind::ColumnTranche},` with an optional
 *  trailing dependency argument this does not care about. */
function parseRegistryHeader(source: string): Map<string, NimbusFieldKind> {
  const out = new Map<string, NimbusFieldKind>();
  const row = /\{"([a-z0-9_]+)",\s*"[a-z0-9_{}]*",\s*OutputFieldKind::(\w+)/g;
  for (const [, fieldId, enumerator] of source.matchAll(row)) {
    const kind = KIND_BY_ENUMERATOR[enumerator];
    if (!kind) throw new Error(`unmapped OutputFieldKind::${enumerator} for '${fieldId}'`);
    const existing = out.get(fieldId);
    if (existing && existing !== kind) {
      throw new Error(`'${fieldId}' declared as both ${existing} and ${kind}`);
    }
    out.set(fieldId, kind);
  }
  return out;
}

/**
 * Parse one preset's field list out of output_config.cpp.
 *
 * The list is a brace-initialized `std::vector<std::string>` with interleaved
 * `//` comments, so the quoted names are extracted rather than the whole
 * initializer parsed. Anchoring on the exact variable name matters: matching
 * loosely would silently read a neighbouring preset and the mirror check would
 * pass against the wrong list.
 */
function parsePresetFieldList(source: string, variable: string): Set<string> {
  const declaration = new RegExp(`${variable}\\s*=\\s*\\{([\\s\\S]*?)\\};`);
  const body = declaration.exec(source);
  if (!body) throw new Error(`no '${variable}' initializer in ${OUTPUT_CONFIG_SOURCE}`);
  const withoutComments = body[1].replace(/\/\/[^\n]*/g, "");
  return new Set([...withoutComments.matchAll(/"([a-z0-9_]+)"/g)].map(([, name]) => name));
}

/** Every field name any plot type or preset can name in a readiness verdict. */
function gatingFieldNames(): string[] {
  const names = new Set<string>();
  const add = (req: FieldReq): void => {
    if (req === "anyField") return;
    for (const clause of req) for (const name of clause) names.add(name);
  };
  for (const meta of PLOT_TYPE_META) add(meta.requires);
  for (const preset of PLOT_PRESETS) {
    add(presetRequirement(preset));
    names.add(preset.field);
  }
  return [...names].sort();
}

// ---------- 1. The mirror agrees with the engine header ----------

check("1. engine registry mirror", () => {
  if (!existsSync(REGISTRY_HEADER)) {
    // Not a pass. The UI can be checked out without the C++ tree, but a silent
    // skip is how a mirror rots, so say so loudly and fail the drift check.
    failed++;
    failures.push(
      `SKIPPED-AS-FAILURE engine header not found at ${REGISTRY_HEADER};`
      + " NIMBUS_OUTPUT_FIELDS could not be verified against the engine",
    );
    return;
  }
  const engine = parseRegistryHeader(readFileSync(REGISTRY_HEADER, "utf8"));
  assertTrue(engine.size > 0, "parsed at least one descriptor from the header");

  const missingFromMirror = [...engine.keys()].filter((id) => !(id in NIMBUS_OUTPUT_FIELDS));
  assertTrue(
    missingFromMirror.length === 0,
    `every engine field is mirrored (missing: ${missingFromMirror.join(", ") || "none"})`,
  );

  const inventedInMirror = Object.keys(NIMBUS_OUTPUT_FIELDS).filter((id) => !engine.has(id));
  assertTrue(
    inventedInMirror.length === 0,
    `the mirror invents no fields (extra: ${inventedInMirror.join(", ") || "none"})`,
  );

  const wrongKind = [...engine].filter(([id, kind]) => NIMBUS_OUTPUT_FIELDS[id] !== kind);
  assertTrue(
    wrongKind.length === 0,
    "every mirrored shape class matches the engine ("
    + (wrongKind.map(([id, kind]) => `${id}: engine ${kind}, mirror ${NIMBUS_OUTPUT_FIELDS[id]}`).join("; ") || "all agree")
    + ")",
  );
});

// ---------- 1b. The preset mirror agrees with the engine ----------

/*
 * `STUDIO_CATALOG_PRESET_FIELDS` is the second hand-carried copy, and it fails
 * quietly in both directions. Drop a field the engine added and the panel sends
 * a user to `studio_full` for something it does not publish — they rerun a
 * multi-hour job and the view is still dark. Invent one and the panel names the
 * expensive preset for a field the cheap one already covers.
 */
check("1b. studio_full preset mirror", () => {
  if (!existsSync(OUTPUT_CONFIG_SOURCE)) {
    failed++;
    failures.push(
      `SKIPPED-AS-FAILURE engine output config not found at ${OUTPUT_CONFIG_SOURCE};`
      + " STUDIO_CATALOG_PRESET_FIELDS could not be verified against the engine",
    );
    return;
  }
  const engine = parsePresetFieldList(
    readFileSync(OUTPUT_CONFIG_SOURCE, "utf8"),
    "k_studio_full_fields",
  );
  assertTrue(engine.size > 0, "parsed the studio_full field list from the engine");

  const missingFromMirror = [...engine].filter((id) => !STUDIO_CATALOG_PRESET_FIELDS.has(id));
  assertTrue(
    missingFromMirror.length === 0,
    `every studio_full field is mirrored (missing: ${missingFromMirror.join(", ") || "none"})`,
  );

  const inventedInMirror = [...STUDIO_CATALOG_PRESET_FIELDS].filter((id) => !engine.has(id));
  assertTrue(
    inventedInMirror.length === 0,
    `the preset mirror invents no fields (extra: ${inventedInMirror.join(", ") || "none"})`,
  );

  // A preset that names a field the registry does not define would be a run
  // that silently writes less than it was asked for.
  const unregistered = [...engine].filter((id) => nimbusFieldKind(id) == null);
  assertTrue(
    unregistered.length === 0,
    `every studio_full field is a registry field (unregistered: ${unregistered.join(", ") || "none"})`,
  );

  // The engine documents studio_full as a superset of presentation. If that
  // ever stops holding, recommending studio_full could *remove* a field the
  // user already had.
  const presentation = parsePresetFieldList(
    readFileSync(OUTPUT_CONFIG_SOURCE, "utf8"),
    "k_presentation_fields",
  );
  const droppedFromPresentation = [...presentation].filter((id) => !engine.has(id));
  assertTrue(
    presentation.size > 0 && droppedFromPresentation.length === 0,
    `studio_full is a superset of presentation (dropped: ${droppedFromPresentation.join(", ") || "none"})`,
  );
});

// ---------- 1c. The recommendation is the narrowest covering preset ----------

check("1c. preset recommendation picks the narrowest covering value", () => {
  for (const field of ["stretching_term", "qc", "divergence", "precip_rate", "vorticity_r"]) {
    assertTrue(
      presetPublishing(field) === STUDIO_CATALOG_PRESET,
      `'${field}' is covered by ${STUDIO_CATALOG_PRESET} (it is in the mirrored list)`,
    );
  }
  // Real registry fields deliberately left outside the Studio catalog: nothing
  // in the view catalog reads them, so they must not drag the recommendation up.
  for (const field of ["zdr", "ppi_sweep", "okubo_weiss", "angular_momentum"]) {
    assertTrue(
      nimbusFieldKind(field) != null,
      `'${field}' is a real engine field (the premise of this case)`,
    );
    assertTrue(
      presetPublishing(field) === WIDEST_OUTPUT_PRESET,
      `'${field}' needs ${WIDEST_OUTPUT_PRESET} (it is outside ${STUDIO_CATALOG_PRESET})`,
    );
  }
  // Deliberately a name the engine has never had, rather than a real CM1-only
  // product: the CM1-only set is empty against this engine (section 3b), and an
  // exemplar picked from it would make this case vacuous the moment the engine
  // absorbed it — which is exactly what happened to the four it used to hold.
  const absent = "not_a_registry_field";
  assertTrue(
    nimbusFieldKind(absent) == null,
    `'${absent}' is not a registry field (the premise of this case)`,
  );
  assertTrue(
    presetPublishing(absent) == null,
    "a field the engine cannot emit has no covering preset at all",
  );
  assertTrue(
    presetPublishing(TRAJECTORY_CAPABILITY) == null,
    "the trajectory capability is not a field and has no covering preset",
  );

  assertTrue(
    presetPublishingAll(["qc", "divergence"]) === STUDIO_CATALOG_PRESET,
    "a group wholly inside the catalog asks only for the catalog preset",
  );
  assertTrue(
    presetPublishingAll(["qc", "zdr"]) === WIDEST_OUTPUT_PRESET,
    "one uncovered member escalates the whole group",
  );
  assertTrue(
    presetPublishingAll(["qc", "potential_vorticity"]) === STUDIO_CATALOG_PRESET,
    "a member no preset can publish does not escalate the group it cannot join",
  );
  assertTrue(
    presetPublishingAll([]) === STUDIO_CATALOG_PRESET,
    "an empty group does not recommend the widest preset",
  );
});

// ---------- 2. Provenance is total over what gating can name ----------

check("2. every gating field resolves to some origin", () => {
  for (const field of gatingFieldNames()) {
    assertTrue(
      fieldOrigins(field).length > 0,
      `'${field}' has at least one origin (a gating field with none can never turn on)`,
    );
  }
});

check("2b. the trajectory sentinel is an artifact, not a field", () => {
  assertTrue(
    fieldOrigins(TRAJECTORY_CAPABILITY).join(",") === "run_artifact",
    "the trajectory capability resolves only to run_artifact",
  );
  assertTrue(
    !(TRAJECTORY_CAPABILITY in NIMBUS_OUTPUT_FIELDS),
    "the trajectory sentinel is not claimed as an output field",
  );
});

/*
 * Trajectories were CLI-only (`--trajectories`), and Studio's runner passes a
 * fixed argument list, so the advice could not say anything a Studio user could
 * act on. The engine now takes `output.trajectories`, and the advice has to name
 * that key rather than gesture at "enable trajectory output" — and must not
 * offer a preset, because no preset can produce a run-level artifact.
 */
check("2c. trajectory advice names the config key, not a preset", () => {
  const remedy = describeMissingFieldRemedy(TRAJECTORY_CAPABILITY, "nimbus_run");
  const action = describeRemedyGroup("trajectory_artifact", "nimbus_run", STUDIO_CATALOG_PRESET).action;
  for (const [label, text] of [["field remedy", remedy], ["group action", action]] as const) {
    assertTrue(
      /output\.trajectories/.test(text),
      `the trajectory ${label} names output.trajectories — got "${text}"`,
    );
    assertTrue(
      !new RegExp(`preset to (${STUDIO_CATALOG_PRESET}|${WIDEST_OUTPUT_PRESET})`).test(text),
      `the trajectory ${label} does not prescribe a preset that cannot help — got "${text}"`,
    );
  }
  const cm1 = describeMissingFieldRemedy(TRAJECTORY_CAPABILITY, "cm1_netcdf");
  assertTrue(
    !/output\.trajectories/.test(cm1),
    `a CM1 dataset is not told to set a Nimbus run key — got "${cm1}"`,
  );
});

// ---------- 3. The impossible-advice fields ----------

/*
 * This is the reason this module exists: a product a plot can require, which a
 * CM1 history writes and the Nimbus engine cannot emit under any configuration.
 * On such a field the old panel told a Nimbus user to add it to the output
 * selection and rerun — hours of compute for a field that was never going to
 * appear.
 *
 * The set is derived, never listed. It has already been four members
 * (potential_vorticity, precipitable_water, liquid_water_path,
 * cloud_water_path), then one, and as of the engine this ran against it is
 * empty: every canonical name in Studio's CM1 crosswalk now resolves to a
 * registry field. A frozen list would have gone on asserting a premise the
 * engine had stopped satisfying, which is the failure mode section 1 exists to
 * catch — so it is not repeated here.
 */
function cm1OnlyGatingFields(): string[] {
  return gatingFieldNames().filter((field) => fieldIsForeignToSource(field, "nimbus_run"));
}

check("3. CM1-only fields are foreign to a Nimbus run", () => {
  for (const field of cm1OnlyGatingFields()) {
    assertTrue(
      !(field in NIMBUS_OUTPUT_FIELDS),
      `'${field}' is absent from the engine registry (the premise of this case)`,
    );
    assertTrue(
      fieldOrigins(field).includes("cm1_history"),
      `'${field}' is reachable from a CM1 history`,
    );
    const remedy = describeMissingFieldRemedy(field, "nimbus_run");
    assertTrue(
      !/rerun/i.test(remedy),
      `'${field}' advice does not ask for a rerun that cannot help — got "${remedy}"`,
    );
    assertTrue(
      /CM1/.test(remedy),
      `'${field}' advice names CM1 as the actual source — got "${remedy}"`,
    );
    assertTrue(
      summarizeMissingFieldReason(field, "nimbus_run") === "CM1 datasets only",
      `'${field}' is chipped as CM1-only on a Nimbus run`,
    );
  }
});

/*
 * The loop above is vacuous whenever the engine covers the whole crosswalk, and
 * a quietly vacuous test is indistinguishable from a passing one. So the state
 * is asserted explicitly, and the group's copy is pinned separately — the
 * classification has to stay correct through the window where no field happens
 * to land in it, because that is precisely when nobody is looking at it.
 */
check("3b. the CM1-only classification is derived, and its state is stated", () => {
  const cm1Only = cm1OnlyGatingFields();
  const uncovered = gatingFieldNames().filter((field) => fieldOrigins(field).length === 0);
  assertTrue(
    uncovered.length === 0,
    `no gating field is unknown to every catalog (unknown: ${uncovered.join(", ") || "none"})`,
  );
  // Not an assertion about which state is right — either is legitimate. It
  // fails only if the two ways of asking disagree.
  assertTrue(
    cm1Only.every((field) => nimbusFieldKind(field) == null),
    `every CM1-only field is genuinely absent from the registry (set: ${cm1Only.join(", ") || "empty"})`,
  );

  const { title, action } = describeRemedyGroup("inapplicable", "nimbus_run", STUDIO_CATALOG_PRESET);
  assertTrue(
    /Not producible from a Nimbus run/.test(title),
    `the inapplicable group is named for what it is — got "${title}"`,
  );
  assertTrue(
    /no output selection or rerun can produce them/.test(action) && /CM1/.test(action),
    `the inapplicable action rules out the rerun and names the real source — got "${action}"`,
  );
});

// ---------- 4. The mirror-image defect, on CM1 datasets ----------

/*
 * The same bug pointed the other way: the CM1 guidance said to "open a history
 * set that wrote the product", but CM1 writes no equivalent of these at all.
 *
 * Derived for the same reason as section 3 — the crosswalk gains variables — but
 * unlike section 3 this direction is not close to empty, so the count is
 * asserted to keep a crosswalk change from silently emptying the case.
 */
const NIMBUS_ONLY_FIELDS = gatingFieldNames()
  .filter((field) => fieldIsForeignToSource(field, "cm1_netcdf"));

check("4. Nimbus-only fields are foreign to a CM1 dataset", () => {
  assertTrue(
    NIMBUS_ONLY_FIELDS.length > 0,
    `some gating field is Nimbus-only, or this case proves nothing (found ${NIMBUS_ONLY_FIELDS.length})`,
  );
  for (const field of NIMBUS_ONLY_FIELDS) {
    assertTrue(
      field in NIMBUS_OUTPUT_FIELDS,
      `'${field}' is a real engine field (the premise of this case)`,
    );
    assertTrue(
      !fieldOrigins(field).includes("cm1_history"),
      `'${field}' has no CM1 mapping`,
    );
    assertTrue(
      fieldIsForeignToSource(field, "cm1_netcdf"),
      `'${field}' is reported as foreign to a CM1 dataset`,
    );
    const remedy = describeMissingFieldRemedy(field, "cm1_netcdf");
    assertTrue(
      !/open a history set/i.test(remedy),
      `'${field}' advice does not ask the user to open a history set that cannot exist — got "${remedy}"`,
    );
    assertTrue(
      summarizeMissingFieldReason(field, "cm1_netcdf") === "Nimbus runs only",
      `'${field}' is chipped as Nimbus-only on a CM1 dataset`,
    );
  }
});

// ---------- 5. Fields the engine does have keep actionable advice ----------

/*
 * The advice names a config key, so the key has to exist.
 *
 * This nearly shipped wrong: `output.fields` accepts a per-field list, but only
 * on the v0 path, where the migrator renames it to `output.preset`. The v1
 * loader Studio writes for reads `output.preset` as an enum, so "add the field
 * to output.fields" would have been YAML the loader rejects — the same shape of
 * impossible instruction this module exists to remove, reintroduced by the fix.
 */
check("5a. every key the advice names exists in the schema Studio writes", () => {
  const output = SCHEMA_SECTIONS.find((section) => section.id === "output");
  assertTrue(output != null, "the config schema has an output section");
  const preset = output?.fields.find((field) => field.key === "preset");
  assertTrue(preset != null, "the output section has a preset field");
  const values = preset && preset.kind === "enum"
    ? preset.options.map((option) => option.value)
    : [];
  for (const recommended of [STUDIO_CATALOG_PRESET, WIDEST_OUTPUT_PRESET]) {
    assertTrue(
      values.includes(recommended),
      `'${recommended}' is an accepted output.preset value (schema offers ${values.join(", ")})`,
    );
  }
  assertTrue(
    output?.fields.every((field) => field.key !== "fields") === true,
    "the schema has no output.fields key, so no advice may name one",
  );
  // The trajectory advice names three keys; a user told to set one that the
  // editor cannot show has to go hand-edit YAML to find out whether it exists.
  for (const key of ["trajectories", "trajectory_seed_time_s", "trajectory_count"]) {
    assertTrue(
      output?.fields.some((field) => field.key === key) === true,
      `the schema exposes output.${key}`,
    );
  }
});

check("5. engine fields keep a concrete, correct remedy", () => {
  for (const field of ["qc", "stretching_term", "precip_rate", "column_max_w", "zdr"]) {
    const remedy = describeMissingFieldRemedy(field, "nimbus_run");
    const covering = presetPublishing(field);
    assertTrue(
      covering != null && /output\.preset/.test(remedy) && remedy.includes(covering),
      `'${field}' advice names the config key and its narrowest covering preset — got "${remedy}"`,
    );
    assertTrue(
      covering === STUDIO_CATALOG_PRESET
        ? !remedy.includes(WIDEST_OUTPUT_PRESET)
        : true,
      `'${field}' advice does not also name the expensive preset it does not need — got "${remedy}"`,
    );
    assertTrue(
      !/output\.fields/.test(remedy),
      `'${field}' advice does not name the v0-only output.fields key — got "${remedy}"`,
    );
    assertTrue(
      !fieldIsForeignToSource(field, "nimbus_run"),
      `'${field}' is not foreign to a Nimbus run`,
    );
    assertTrue(
      summarizeMissingFieldReason(field, "nimbus_run") === "Not in this run's output",
      `'${field}' is chipped as an output-selection gap`,
    );
  }
});

/*
 * The shape wording is checked because it is the part a reader acts on: a user
 * told a product is single-level knows not to look for it on the 3D tab. It
 * describes rank rather than physics on purpose — the engine's Core3D tranche
 * holds diagnostics as well as prognostic state, and its ColumnTranche mixes
 * integrals, maxima, and levels, so any physics word would be false for some
 * member of the class.
 */
check("5b. shape class reaches the remedy text", () => {
  for (const field of ["qc", "stretching_term", "divergence"]) {
    assertTrue(
      /3-D/.test(describeMissingFieldRemedy(field, "nimbus_run")),
      `'${field}' is described as a 3-D field`,
    );
  }
  for (const field of ["column_max_w", "cape", "precip_rate", "ppi_sweep"]) {
    assertTrue(
      /single-level/.test(describeMissingFieldRemedy(field, "nimbus_run")),
      `'${field}' is described as single-level`,
    );
  }
  assertTrue(
    !/prognostic/.test(describeMissingFieldRemedy("stretching_term", "nimbus_run")),
    "a Core3D diagnostic is not called prognostic",
  );
  assertTrue(
    !/integrated/.test(describeMissingFieldRemedy("column_max_w", "nimbus_run")),
    "a column maximum is not called an integral",
  );
});

/*
 * The group heading is the sentence most users act on — it is stated once above
 * the whole list, so it is the one place a wrong preset is cheapest to write and
 * most expensive to follow. The old copy said the widest preset was "the only
 * one that contains every registry field", which is true and was still the wrong
 * thing to tell someone whose missing products all sit in the narrower preset.
 */
check("5c. the group action prescribes the group's own covering preset", () => {
  const studio = describeRemedyGroup("exportable", "nimbus_run", STUDIO_CATALOG_PRESET).action;
  assertTrue(
    new RegExp(`output\\.preset to ${STUDIO_CATALOG_PRESET}`).test(studio),
    `a wholly-covered group is sent to ${STUDIO_CATALOG_PRESET} — got "${studio}"`,
  );
  assertTrue(
    !new RegExp(`output\\.preset to ${WIDEST_OUTPUT_PRESET}`).test(studio),
    `a wholly-covered group is not also sent to ${WIDEST_OUTPUT_PRESET} — got "${studio}"`,
  );

  const widest = describeRemedyGroup("exportable", "nimbus_run", WIDEST_OUTPUT_PRESET).action;
  assertTrue(
    new RegExp(`output\\.preset to ${WIDEST_OUTPUT_PRESET}`).test(widest),
    `an escalated group is sent to ${WIDEST_OUTPUT_PRESET} — got "${widest}"`,
  );
  assertTrue(
    widest.includes(STUDIO_CATALOG_PRESET),
    `an escalated group says why the cheaper preset is not enough — got "${widest}"`,
  );

  // The preset argument is meaningless to the other groups; it must not leak
  // into their copy as a suggestion the user cannot act on.
  for (const group of ["inapplicable", "trajectory_artifact", "no_readable_fields"] as const) {
    for (const preset of [STUDIO_CATALOG_PRESET, WIDEST_OUTPUT_PRESET] as const) {
      const { action } = describeRemedyGroup(group, "nimbus_run", preset);
      assertTrue(
        !/output\.preset to /.test(action),
        `the ${group} action prescribes no preset — got "${action}"`,
      );
    }
  }

  // A CM1 dataset has no Nimbus preset to set, at any coverage.
  for (const preset of [STUDIO_CATALOG_PRESET, WIDEST_OUTPUT_PRESET] as const) {
    const { action } = describeRemedyGroup("exportable", "cm1_netcdf", preset);
    assertTrue(
      !/output\.preset/.test(action),
      `CM1 guidance names no Nimbus output preset — got "${action}"`,
    );
  }
});

/*
 * The split that section 5c's two headings exist for.
 *
 * A single exportable group has to name one preset for all its members, so one
 * blocked product outside the Studio catalog would push every product the
 * cheaper preset covers onto `dynamics_full` — the over-broad advice this module
 * was written to delete, arriving through the fix for it. A browser fixture
 * caught exactly that, which is why the case is pinned here where it costs a
 * second to run.
 */
check("5d. one out-of-catalog product does not escalate the rest", () => {
  const advice = ["qc", "stretching_term", "divergence", "zdr", "ppi_sweep"]
    .map((field) => describeMissingClause([field], "nimbus_run"));
  const sections = groupRemedyAdvice(advice, (item) => item, "nimbus_run");
  const exportable = sections.filter(({ group }) => group === "exportable");
  assertTrue(
    exportable.length === 2,
    `a mixed group splits by covering preset (got ${exportable.length} exportable sections)`,
  );

  const cheap = exportable.find(({ preset }) => preset === STUDIO_CATALOG_PRESET);
  const dear = exportable.find(({ preset }) => preset === WIDEST_OUTPUT_PRESET);
  assertTrue(
    cheap?.entries.map(({ subject }) => subject).join(",") === "qc,stretching_term,divergence",
    `the catalog products stay on ${STUDIO_CATALOG_PRESET} (got ${cheap?.entries.map(({ subject }) => subject).join(",")})`,
  );
  assertTrue(
    dear?.entries.map(({ subject }) => subject).join(",") === "zdr,ppi_sweep",
    `only the out-of-catalog products ask for ${WIDEST_OUTPUT_PRESET} (got ${dear?.entries.map(({ subject }) => subject).join(",")})`,
  );
  assertTrue(
    sections.findIndex(({ preset }) => preset === STUDIO_CATALOG_PRESET)
      < sections.findIndex(({ preset }) => preset === WIDEST_OUTPUT_PRESET),
    "the cheaper action is stated first",
  );
  assertTrue(
    new Set(sections.map(({ key }) => key)).size === sections.length,
    "each section has a distinct render key, so a repeated group does not collide",
  );
  assertTrue(
    cheap?.title !== dear?.title,
    "the two exportable sections do not share a heading a reader cannot tell apart",
  );

  // A group with nothing out of catalog stays whole; the split is a response to
  // the data, not a permanent second heading.
  const uniform = groupRemedyAdvice(
    ["qc", "divergence"].map((field) => describeMissingClause([field], "nimbus_run")),
    (item) => item,
    "nimbus_run",
  );
  assertTrue(
    uniform.filter(({ group }) => group === "exportable").length === 1,
    "a wholly-covered group renders as one section",
  );

  // CM1 guidance names no preset, so splitting it would print the same sentence
  // twice under the same heading.
  const cm1 = groupRemedyAdvice(
    ["qc", "zdr"].map((field) => describeMissingClause([field], "cm1_netcdf")),
    (item) => item,
    "cm1_netcdf",
  );
  assertTrue(
    cm1.filter(({ group }) => group === "exportable").length <= 1,
    "a CM1 dataset never splits the exportable group",
  );
});

// ---------- 6. Studio-derived fields are never called missing-with-a-rerun ----------

/*
 * dewpoint, relative humidity, and theta_e are computed by Studio from theta,
 * p, and qv. A run that publishes those dependencies already offers them, so
 * they only appear blocked when a dependency is genuinely absent — and telling
 * that user to re-export "dewpoint" would be advice for the wrong field.
 */
check("6. Studio-derived fields are not treated as foreign", () => {
  for (const field of ["dewpoint", "relative_humidity", "theta_e", "temperature"]) {
    assertTrue(
      fieldOrigins(field).includes("studio_derived"),
      `'${field}' is recognized as Studio-derived`,
    );
    for (const source of ["nimbus_run", "cm1_netcdf"] as const) {
      assertTrue(
        !fieldIsForeignToSource(field, source),
        `'${field}' is never foreign on ${source} — Studio can compute it`,
      );
    }
  }
});

// ---------- 7. No remedy is empty or self-contradicting ----------

check("7. every gating field yields a non-empty remedy on both sources", () => {
  for (const field of gatingFieldNames()) {
    for (const source of ["nimbus_run", "cm1_netcdf"] as const) {
      const remedy = describeMissingFieldRemedy(field, source);
      assertTrue(remedy.length > 0, `'${field}' has a remedy on ${source}`);
      assertTrue(
        !/Studio has no source/.test(remedy),
        `'${field}' does not fall through to the no-source branch on ${source}`,
      );
      assertTrue(
        summarizeMissingFieldReason(field, source).length > 0,
        `'${field}' has a reason chip on ${source}`,
      );
    }
  }
});

// ---------- 8. Inherited object keys are not mistaken for engine fields ----------

/*
 * The mirror is an object literal, so a bare `field in NIMBUS_OUTPUT_FIELDS`
 * would answer true for every Object.prototype member. Field names reach these
 * functions from requirement tables and, on the CM1 path, from netCDF variable
 * names in a file — so a variable called `constructor` would have been reported
 * as a real engine field and given export-and-rerun advice.
 */
check("8. prototype members are not engine fields", () => {
  for (const name of ["constructor", "toString", "hasOwnProperty", "__proto__", "valueOf"]) {
    assertTrue(
      nimbusFieldKind(name) === null,
      `'${name}' is not reported as an engine field`,
    );
    assertTrue(
      !fieldOrigins(name).includes("nimbus_output"),
      `'${name}' does not claim a Nimbus output origin`,
    );
  }
});

// ---------- 9. The per-product detail is well formed ----------

/*
 * `detail` is concatenated from optional parts, so an absent part must not
 * leave a dangling separator on either end — the kind of thing that reads as a
 * rendering bug and undermines every sentence next to it.
 */
check("9. detail carries no dangling separators", () => {
  for (const field of gatingFieldNames()) {
    for (const source of ["nimbus_run", "cm1_netcdf"] as const) {
      const { detail } = describeMissingClause([field], source);
      assertTrue(
        detail === detail.trim() && !detail.startsWith("·") && !detail.endsWith("·"),
        `'${field}' on ${source} has a clean detail — got "${detail}"`,
      );
    }
  }
});

check("9b. an OR clause names its alternatives once", () => {
  const { detail, fields } = describeMissingClause(
    ["cloud_condensate", "total_condensate"],
    "nimbus_run",
  );
  assertTrue(fields.length === 2, "both alternatives are carried through");
  assertTrue(
    detail.includes("any one of cloud_condensate, total_condensate will do"),
    `an OR clause says either will do — got "${detail}"`,
  );
  assertTrue(
    detail.split("any one of").length === 2,
    `the alternatives are stated once, not repeated — got "${detail}"`,
  );
});

// ---------- Which fields a 3-D renderer can draw ----------

check("6a. volume-shaped engine fields are offered to the 3D renderers", () => {
  for (const field of ["w", "u", "v", "theta", "qv", "reflectivity_dbz", "vorticity_z", "p"]) {
    assertTrue(isVolumetricField(field), `${field} is volumetric`);
  }
});

check("6b. single-level products are withheld from the 3D renderers", () => {
  /*
   * These carry no vertical dimension, so marching cubes and the volume ray-march have
   * nothing to work on. Offering them reported a shape error only after the load; on a
   * beast_xl run 16 of 37 published fields were dead options in the 3D field menu.
   */
  for (const field of [
    "cape", "cin", "lcl", "lfc", "el", "vil", "stp", "pbl_height",
    "composite_reflectivity", "updraft_helicity", "srh_0_1km", "srh_0_3km",
    "cloud_base_height", "cloud_top_height",
    "surface_latent_heat_flux", "surface_sensible_heat_flux",
  ]) {
    assertTrue(!isVolumetricField(field), `${field} is single-level`);
  }
});

check("6c. Studio's derived volume fields stay available in 3D", () => {
  for (const field of ["dewpoint", "relative_humidity", "theta_e"]) {
    assertTrue(isVolumetricField(field), `${field} is volumetric`);
  }
});

check("6d. an unclassifiable field is offered rather than hidden", () => {
  /*
   * The catalog covers engine and Studio-derived output only. User expressions, external
   * providers, q_criterion and the in-progress Cartesian vorticity_x/y names are absent,
   * and every one of them is volumetric — so permissive is the correct default. Hiding a
   * field Studio cannot classify would remove data the user actually has.
   */
  for (const field of ["q_criterion", "vorticity_x", "horizontal_wind_speed", "my_expression"]) {
    assertTrue(isVolumetricField(field), `${field} is offered`);
  }
});

check("6e. the filter preserves order and drops only single-level fields", () => {
  const kept = volumetricFields(["w", "cape", "theta", "vil", "vorticity_z"]);
  assertTrue(kept.join(",") === "w,theta,vorticity_z", "keeps volume fields in order");
  assertTrue(volumetricFields([]).length === 0, "an empty list stays empty");
});

check("6f. every catalog field is classified consistently with its declared kind", () => {
  // Guards the predicate drifting into a second hand-maintained list of names.
  for (const [field, kind] of Object.entries(NIMBUS_OUTPUT_FIELDS)) {
    const expected = kind === "core_volume" || kind === "derived_volume";
    assertTrue(isVolumetricField(field) === expected, `${field} (${kind}) matches its kind`);
  }
});

check("7a. lossless or legacy fields stay silent", () => {
  const step = { fieldCompression: {} } as StepInfo;
  assertTrue(compressionDisclosures(step, ["w"]).length === 0, "empty contract is silent");
  assertTrue(compressionDisclosures(null, ["w"]).length === 0, "missing step is silent");
});

check("7b. identical selected-field compression contracts group without duplication", () => {
  const contract = {
    codec: "zfp",
    mode: "accuracy",
    modeParameter: 1e-5,
    modeParameterUnits: "kg/kg",
    errorContract: "absolute decoded error <= parameter",
    sparseThreshold: 0,
    float16Prefilter: false,
    decodedDtype: "float32",
  };
  const step = { fieldCompression: { qc: contract, qi: contract } } as unknown as StepInfo;
  const disclosures = compressionDisclosures(step, ["qi", "qc", "qi", null]);
  assertTrue(disclosures.length === 1, "identical contracts form one disclosure");
  assertTrue(disclosures[0]?.fields.join(",") === "qc,qi", "fields are unique and stable");
  assertTrue(disclosures[0]?.shortLabel.includes("ZFP accuracy"), "codec and mode are visible");
  assertTrue(disclosures[0]?.detail.includes("sparse threshold off"), "sparse policy is explicit");
  assertTrue(disclosures[0]?.detail.includes("float16 prefilter off"), "prefilter policy is explicit");
});

check("7c. materially different contracts remain separate", () => {
  const base = {
    codec: "zfp",
    mode: "accuracy",
    modeParameter: 1e-5,
    modeParameterUnits: "1/s",
    errorContract: "absolute",
    sparseThreshold: 0,
    float16Prefilter: false,
    decodedDtype: "float32",
  };
  const step = {
    fieldCompression: {
      vorticity_z: base,
      w: { ...base, modeParameter: 1e-3, sparseThreshold: 1e-4, float16Prefilter: true },
    },
  } as unknown as StepInfo;
  const disclosures = compressionDisclosures(step, ["w", "vorticity_z"]);
  assertTrue(disclosures.length === 2, "different contracts are not collapsed");
  assertTrue(disclosures.some((entry) => entry.detail.includes("float16 prefilter on")), "lossy prefilter is disclosed");
});

check("7d. temporal products retain every distinct per-frame contract", () => {
  const contract = {
    codec: "zfp",
    mode: "accuracy",
    modeParameter: 1e-5,
    modeParameterUnits: "m/s",
    errorContract: "absolute",
    sparseThreshold: 0,
    float16Prefilter: false,
    decodedDtype: "float32",
  };
  const steps = [
    { index: 4, fieldCompression: { w: contract } },
    { index: 5, fieldCompression: { w: contract } },
    { index: 6, fieldCompression: { w: { ...contract, modeParameter: 2e-5 } } },
  ] as unknown as StepInfo[];
  const disclosures = compressionDisclosuresAcrossSteps(steps, ["w"]);
  assertTrue(disclosures.length === 2, "mixed temporal contracts remain visible");
  assertTrue(disclosures.some((entry) => entry.stepIndices?.join(",") === "4,5"), "matching frame contracts group");
  assertTrue(disclosures.some((entry) => entry.stepIndices?.join(",") === "6"), "changed contract keeps its frame identity");
});

// ---------- Report ----------

console.log(`\nfield provenance tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  " + f);
  process.exit(1);
}
process.exit(0);
