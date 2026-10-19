import { parse, stringify } from "yaml";
import type { ConfigObject } from "./configSchema";

// YAML parse/stringify with explicit options matching tests/fixtures/configs/test/*.yaml formatting.

const STRINGIFY_OPTIONS = {
  indent: 2,
  lineWidth: 0, // never wrap long lines
  doubleQuotedAsJSON: false,
  defaultStringType: "PLAIN" as const,
};

export function parseYaml(text: string): ConfigObject {
  const parsed = parse(text) ?? {};
  if (typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Top-level YAML must be an object");
  }
  return parsed as ConfigObject;
}

export function stringifyYaml(obj: ConfigObject): string {
  return stringify(obj, STRINGIFY_OPTIONS);
}

/**
 * Deep merge `template` into `defaults`. Used when loading a template into
 * a fresh defaults-populated config so missing fields fall back to schema
 * defaults rather than `undefined`.
 */
export function mergeConfig(
  defaults: ConfigObject,
  template: ConfigObject,
): ConfigObject {
  const result: ConfigObject = { ...defaults };
  for (const [key, value] of Object.entries(template)) {
    const existing = result[key];
    if (
      value !== null &&
      typeof value === "object" &&
      !Array.isArray(value) &&
      existing !== null &&
      typeof existing === "object" &&
      !Array.isArray(existing)
    ) {
      result[key] = mergeConfig(existing as ConfigObject, value as ConfigObject);
    } else {
      result[key] = value;
    }
  }
  return result;
}
