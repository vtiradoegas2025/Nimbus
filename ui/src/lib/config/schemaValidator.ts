/* v1 schema-version validator. Gate between parseYaml() and mergeConfig().
 * v0 configs need migration; undeclared/unknown stamp is a hard stop. */

import type { ConfigObject } from "./configSchema";

/* ---------- Public types ---------- */

export type SchemaStatus =
  | "v1_valid"
  | "v1_with_warnings"
  | "v1_invalid"
  | "v0_legacy"
  | "stamp_unknown"
  | "stamp_missing"
  | "not_an_object";

export interface SchemaError {
  /** Short stable code; UI can branch on this. */
  code:
    | "stamp_missing"
    | "stamp_not_string"
    | "stamp_unknown"
    | "stamp_v0_legacy"
    | "unknown_section"
    | "section_not_object"
    | "not_an_object";
  /** Human-readable message. */
  message: string;
  /** Optional path (e.g. "boundary_conditions"). */
  path?: string;
}

export type SchemaWarning = SchemaError;

export interface SchemaValidationResult {
  status: SchemaStatus;
  /** The literal value of `schema:` in the file. Null if absent or non-string. */
  schemaStamp: string | null;
  errors: SchemaError[];
  warnings: SchemaWarning[];
}

/* ---------- Constants ---------- */

/**
 * Canonical v1 schema stamp. Configs MUST set `schema: nimbus.v1` at the
 * top level. Mirrors the C++ `schema:` lookup in `v1_loader.cpp`.
 */
export const SCHEMA_V1_STAMP = "nimbus.v1";

/**
 * Top-level v1 section names, per docs/CM1Mapping.md §20. Sections not
 * in this set are flagged as warnings — the user might be hand-editing a
 * v1 config and made a typo, or pasted in a v0 fragment.
 */
export const V1_TOP_LEVEL_KEYS = new Set<string>([
  "schema",
  "grid",
  "time",
  "dynamics",
  // v1_loader.cpp and config_v1_serialize.cpp expose the scalar
  // reconstruction choice as its own top-level section.
  "advection",
  "boundary_conditions",
  "damping",
  "coriolis",
  "microphysics",
  "turbulence",
  "diffusion",
  "surface",
  "boundary_layer",
  "radiation",
  "radar",
  "terrain",
  "initial_conditions",
  // v1_loader.cpp take_domain_motion() reads it and config_v1_serialize.cpp
  // writes it, so `--migrate-config` emits it for every v0 config. The form
  // has no section for it yet; recognizing it here keeps the YAML view from
  // reporting a real section as a typo.
  "domain_motion",
  "output",
  "extensions",
]);

/** v0 tells: keys that were top-level in v0 or under different parents.
 *  Distinguish "needs migration" from "typo in v1 file". */
export const V0_TELL_KEYS = new Set<string>([
  "duration_s",
  "coordinate_system",
  "environment",
  "hodograph",
  "trigger",
  "operator_tier",
  "numerics",       // collapsed into time + diffusion in v1
  "chaos",          // moved under extensions.chaos in v1
  "system",         // moved under extensions in v1
]);

/* ---------- Public API ---------- */

/** Validate schema version: not_an_object → v0_legacy/stamp_missing → stamp check → shape check. */
export function validateSchemaVersion(
  parsed: unknown,
): SchemaValidationResult {
  // --- 1. Type guard ---
  if (
    parsed === null ||
    typeof parsed !== "object" ||
    Array.isArray(parsed)
  ) {
    return {
      status: "not_an_object",
      schemaStamp: null,
      errors: [
        {
          code: "not_an_object",
          message:
            "Top-level YAML must be a mapping (key: value). Got " +
            (Array.isArray(parsed) ? "array" : typeof parsed) +
            ".",
        },
      ],
      warnings: [],
    };
  }

  const config = parsed as ConfigObject;
  const stampRaw = config["schema"];

  // --- 2. Stamp detection ---
  if (stampRaw === undefined || stampRaw === null) {
    const looksLikeV0 = hasV0TellKeys(config);
    if (looksLikeV0) {
      return {
        status: "v0_legacy",
        schemaStamp: null,
        errors: [
          {
            code: "stamp_v0_legacy",
            message:
              "This looks like a Nimbus v0 (pre-CM1-mirror) config. " +
              "It must be migrated to v1 before Studio can edit it. " +
              "Run `bin/nimbus_sim --migrate-config PATH` and re-upload " +
              "the result.",
          },
        ],
        warnings: [],
      };
    }
    return {
      status: "stamp_missing",
      schemaStamp: null,
      errors: [
        {
          code: "stamp_missing",
          message:
            "Missing required top-level field `schema: nimbus.v1`. " +
            "Every Nimbus Studio config must declare its schema version.",
        },
      ],
      warnings: [],
    };
  }

  if (typeof stampRaw !== "string") {
    return {
      status: "stamp_unknown",
      schemaStamp: null,
      errors: [
        {
          code: "stamp_not_string",
          message:
            "Top-level `schema:` must be a string (e.g. `nimbus.v1`). " +
            "Got " +
            typeof stampRaw +
            ".",
        },
      ],
      warnings: [],
    };
  }

  if (stampRaw !== SCHEMA_V1_STAMP) {
    return {
      status: "stamp_unknown",
      schemaStamp: stampRaw,
      errors: [
        {
          code: "stamp_unknown",
          message:
            "Unrecognized schema stamp `" +
            stampRaw +
            "`. Nimbus Studio supports `nimbus.v1` only. " +
            "If this is a future-version config, upgrade Studio; " +
            "otherwise fix the stamp.",
        },
      ],
      warnings: [],
    };
  }

  // --- 3. v1 shape check ---
  const errors: SchemaError[] = [];
  const warnings: SchemaWarning[] = [];

  for (const [key, value] of Object.entries(config)) {
    if (!V1_TOP_LEVEL_KEYS.has(key)) {
      // Unknown top-level key — most likely a v0 fragment or a typo.
      // We warn rather than error so the user can still edit; the
      // simulator's v1_loader will reject it at run time if it's
      // actually invalid.
      warnings.push({
        code: "unknown_section",
        message:
          "Unknown top-level section `" +
          key +
          "`. Recognized v1 sections are: " +
          [...V1_TOP_LEVEL_KEYS].sort().join(", ") +
          ".",
        path: key,
      });
      continue;
    }
    // Every recognized section except `schema:` (which is a string)
    // must be an object.
    if (key === "schema") continue;
    if (
      value === null ||
      typeof value !== "object" ||
      Array.isArray(value)
    ) {
      errors.push({
        code: "section_not_object",
        message:
          "Section `" +
          key +
          "` must be a mapping (key: value). Got " +
          (Array.isArray(value) ? "array" : typeof value) +
          ".",
        path: key,
      });
    }
  }

  let status: SchemaStatus;
  if (errors.length > 0) {
    status = "v1_invalid";
  } else if (warnings.length > 0) {
    status = "v1_with_warnings";
  } else {
    status = "v1_valid";
  }

  return {
    status,
    schemaStamp: stampRaw,
    errors,
    warnings,
  };
}

/**
 * True if the user-facing flow should block on this result. Used by the
 * upload pipeline to decide whether to load the parsed config into state
 * or surface a hard error.
 */
export function isBlocking(result: SchemaValidationResult): boolean {
  return (
    result.status !== "v1_valid" &&
    result.status !== "v1_with_warnings"
  );
}

/**
 * Compose a one-line user-facing summary of the validation outcome.
 * Suitable for an error banner or toast.
 */
export function summarize(result: SchemaValidationResult): string {
  switch (result.status) {
    case "v1_valid":
      return "Schema OK (nimbus.v1).";
    case "v1_with_warnings":
      return (
        "Schema OK (nimbus.v1) with " +
        result.warnings.length +
        " warning" +
        (result.warnings.length === 1 ? "" : "s") +
        "."
      );
    case "v1_invalid":
      return (
        "Schema validation failed: " +
        result.errors.length +
        " error" +
        (result.errors.length === 1 ? "" : "s") +
        "."
      );
    case "v0_legacy":
      return "v0 config detected — needs migration to nimbus.v1.";
    case "stamp_missing":
      return "Missing `schema: nimbus.v1` stamp.";
    case "stamp_unknown":
      return (
        "Unrecognized schema stamp" +
        (result.schemaStamp ? " `" + result.schemaStamp + "`" : "") +
        "."
      );
    case "not_an_object":
      return "YAML root is not a mapping.";
  }
}

/* ---------- Internals ---------- */

function hasV0TellKeys(config: ConfigObject): boolean {
  for (const key of Object.keys(config)) {
    if (V0_TELL_KEYS.has(key)) {
      return true;
    }
  }
  return false;
}
