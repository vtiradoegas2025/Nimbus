import {
  applyFieldChange,
  defaultsForSchema,
  type ConfigObject,
} from "@/lib/config/configSchema";
import {
  authoritativeRuntimeStateFields,
  buildConfigPreflightRequest,
  type ConfigPreflightRequest,
} from "@/lib/config/configPreflight";
import {
  normalizeSchemeSelections,
  type SchemeCatalog,
} from "@/lib/config/schemeCatalog";
import { stringifyYaml } from "@/lib/config/yaml";
import {
  CM1_SMOKE_DURATION_S,
  CM1_SMOKE_OUTPUT_INTERVAL_S,
} from "./cm1GuidedHandoff";

export interface Cm1SoundingProjectDraft {
  schema: "nimbus.studio.cm1-sounding-project-draft";
  schemaVersion: 1;
  sourceScope: "reviewed_sounding_only" | "native_sounding_only";
  sourceMode: Cm1SoundingSourceMode;
  config: ConfigObject;
  yaml: string;
  preflightRequest: ConfigPreflightRequest;
}

export type Cm1SoundingSourceMode = "reviewed_projection" | "native_cm1";

export interface Cm1SoundingProjectOptions {
  sourceMode?: Cm1SoundingSourceMode;
}

/**
 * Build a complete Nimbus baseline around one reviewed projected sounding.
 * No unresolved CM1 namelist control is guessed: every other value comes from
 * the current Nimbus schema defaults and exact executable catalog.
 */
export function buildCm1SoundingProjectDraft(
  soundingPath: string,
  catalog: SchemeCatalog,
  options: Cm1SoundingProjectOptions = {},
): Cm1SoundingProjectDraft {
  if (soundingPath.trim() === "") {
    throw new Error("a saved sounding path is required for the Nimbus project draft");
  }
  let config = defaultsForSchema();
  config = applyFieldChange(config, "initial_conditions.sounding.type", "file");
  config = applyFieldChange(
    config,
    "initial_conditions.sounding.file_path",
    soundingPath,
  );
  // Both supported CM1 handoff artifacts contain an explicit u/v column.
  // Auto is scoped to this generated project; the global form default remains
  // Custom so existing configurations keep their established wind ownership.
  config = applyFieldChange(config, "initial_conditions.hodograph.type", "auto");
  // This handoff is deliberately a short operational smoke test, not an
  // inferred reproduction of the CM1 experiment duration.
  config = applyFieldChange(config, "time.run_time", CM1_SMOKE_DURATION_S);
  config = applyFieldChange(config, "time.output_interval", CM1_SMOKE_OUTPUT_INTERVAL_S);
  config = normalizeSchemeSelections(catalog, config);
  const preflightRequest = buildConfigPreflightRequest(
    config,
    catalog,
    authoritativeRuntimeStateFields(catalog),
  );
  const sourceMode = options.sourceMode ?? "reviewed_projection";
  const native = sourceMode === "native_cm1";
  const preamble = [
    native
      ? "# Nimbus Studio baseline created from an original native CM1 input_sounding."
      : "# Nimbus Studio baseline created from a reviewed CM1 input_sounding projection.",
    "# IMPORT SCOPE: sounding only. Other values are Nimbus defaults, not translated CM1 controls.",
    native
      ? "# NATIVE SOURCE: theta, qv, u, and v are read directly; Nimbus reconstructs its own EOS state."
      : "# PROJECTED SOURCE: reviewed hydrostatic projection; not an exact CM1 model state.",
    `# GUIDED SMOKE: ${CM1_SMOKE_DURATION_S} s total with ${CM1_SMOKE_OUTPUT_INTERVAL_S} s output; extend only after review.`,
    "# Exact executable/catalog compatibility was checked before this file was offered for saving.",
  ].join("\n");
  return {
    schema: "nimbus.studio.cm1-sounding-project-draft",
    schemaVersion: 1,
    sourceScope: native ? "native_sounding_only" : "reviewed_sounding_only",
    sourceMode,
    config,
    yaml: `${preamble}\n${stringifyYaml(config)}`,
    preflightRequest,
  };
}
