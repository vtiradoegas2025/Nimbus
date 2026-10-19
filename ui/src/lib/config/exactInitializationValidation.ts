import { invokeCommand } from "@/lib/platform/appInvoke";
export interface Cm1SoundingProjectValidation {
  schema: "nimbus.studio.cm1-sounding-project-validation";
  schemaVersion: 1;
  executablePath: string;
  summary: string;
  warnings: string[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** Fail closed on malformed native evidence so an IPC mismatch cannot enable save. */
export function validateCm1SoundingProjectEvidence(
  value: unknown,
): Cm1SoundingProjectValidation {
  if (
    !isRecord(value) ||
    value.schema !== "nimbus.studio.cm1-sounding-project-validation" ||
    value.schemaVersion !== 1 ||
    typeof value.executablePath !== "string" ||
    value.executablePath.trim() === "" ||
    typeof value.summary !== "string" ||
    !value.summary.startsWith("[VALIDATE] PASS:") ||
    !Array.isArray(value.warnings) ||
    !value.warnings.every((warning) => typeof warning === "string")
  ) {
    throw new Error("malformed initialization-validation evidence from Nimbus runtime");
  }
  return value as unknown as Cm1SoundingProjectValidation;
}

/**
 * Ask the configured production simulator to strict-load the exact generated
 * YAML and build its real file-sounding column without allocating 3-D fields
 * or running a timestep. This complements, rather than replaces, catalog
 * preflight.
 */
export async function validateCm1SoundingProjectWithBinary(
  configYaml: string,
  expectedCatalogFingerprint: string,
  simulatorPath?: string,
): Promise<Cm1SoundingProjectValidation> {
  if (
    typeof window === "undefined" ||
    !("__TAURI_INTERNALS__" in (window as Window & { __TAURI_INTERNALS__?: unknown }))
  ) {
    throw new Error("exact-binary initialization validation requires Nimbus Studio desktop");
  }
  const response = await invokeCommand<unknown>("validate_cm1_sounding_project", {
    configYaml,
    simulatorPath,
    expectedCatalogFingerprint,
  });
  return validateCm1SoundingProjectEvidence(response);
}
