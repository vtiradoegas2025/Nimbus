export type ConfigDetailLevel = "basic" | "advanced";

const STORAGE_KEY = "nimbus-studio:config-detail:v1";

/** Basic keeps the decisions needed to define, initialize, execute, publish,
 * and continue an ordinary experiment. Specialist numerics and diagnostics
 * remain in the YAML and reappear unchanged in Advanced. */
export const BASIC_CONFIG_SECTION_IDS = new Set([
  "grid",
  "time",
  "microphysics",
  "turbulence",
  "boundary_layer",
  "radiation",
  "sounding",
  "trigger",
  "output",
  "compute",
  "execution",
  "checkpoint",
]);

export function isConfigDetailLevel(value: unknown): value is ConfigDetailLevel {
  return value === "basic" || value === "advanced";
}

export function sectionsForConfigDetail<T extends { id: string }>(
  sections: readonly T[],
  detail: ConfigDetailLevel,
): T[] {
  return detail === "advanced"
    ? [...sections]
    : sections.filter((section) => BASIC_CONFIG_SECTION_IDS.has(section.id));
}

export function advancedConfigSectionCount<T extends { id: string }>(sections: readonly T[]): number {
  return sections.filter((section) => !BASIC_CONFIG_SECTION_IDS.has(section.id)).length;
}

export function loadConfigDetailLevel(storage?: Pick<Storage, "getItem">): ConfigDetailLevel {
  if (!storage) return "basic";
  try {
    const value = storage.getItem(STORAGE_KEY);
    return isConfigDetailLevel(value) ? value : "basic";
  } catch {
    return "basic";
  }
}

export function saveConfigDetailLevel(
  storage: Pick<Storage, "setItem"> | undefined,
  detail: ConfigDetailLevel,
): void {
  if (!storage) return;
  try {
    storage.setItem(STORAGE_KEY, detail);
  } catch {
    // Presentation preference is best-effort and never affects YAML.
  }
}
