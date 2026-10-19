import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parse as parseYaml } from "yaml";

import {
  authoritativeRuntimeStateFields,
  buildConfigPreflightRequest,
} from "../../../ui/src/lib/config/configPreflight.ts";
import {
  MOCK_SCHEME_CATALOG,
  validateSchemeCatalog,
} from "../../../ui/src/lib/config/schemeCatalog.ts";
import {
  defaultsForSchema,
  reconcileFormConfig,
  type ConfigObject,
} from "../../../ui/src/lib/config/configSchema.ts";
import { mergeConfig } from "../../../ui/src/lib/config/yaml.ts";


const [binaryArgument, manifestArgument, outputArgument, sourceMode] = process.argv.slice(2);
if (sourceMode !== undefined && sourceMode !== "--exact-document") throw new Error("unknown source mode");
if (!binaryArgument || !manifestArgument || !outputArgument) {
  throw new Error(
    "usage: generate_native_parity_batch.ts BINARY MANIFEST_JSON OUTPUT_JSON",
  );
}

const binary = resolve(binaryArgument);
const manifest = JSON.parse(
  readFileSync(resolve(manifestArgument), "utf8"),
) as unknown;
if (
  !Array.isArray(manifest) || manifest.length === 0 ||
  !manifest.every((path) => typeof path === "string" && path.length > 0)
) {
  throw new Error("native parity manifest must contain configuration paths");
}

const catalog = validateSchemeCatalog(JSON.parse(execFileSync(
  binary,
  ["--scheme-catalog=json"],
  { encoding: "utf8", maxBuffer: 32 * 1024 * 1024 },
)));
const browserCatalog = validateSchemeCatalog(MOCK_SCHEME_CATALOG);
const canonicalJson = (value: unknown): string => {
  const canonicalize = (current: unknown): unknown => {
    if (Array.isArray(current)) return current.map(canonicalize);
    if (current !== null && typeof current === "object") {
      return Object.fromEntries(
        Object.entries(current as Record<string, unknown>)
          .sort(([left], [right]) => left.localeCompare(right))
          .map(([key, entry]) => [key, canonicalize(entry)]),
      );
    }
    return current;
  };
  return JSON.stringify(canonicalize(value));
};
if (canonicalJson(catalog.readiness) !== canonicalJson(browserCatalog.readiness)) {
  throw new Error("browser catalog readiness contract drifted from the native catalog");
}
const admissionInteractionPolicy = (policy: object) =>
  Object.fromEntries(
    Object.entries(policy).filter(([key]) => !key.startsWith("qualifiedZfp")),
  );
if (canonicalJson(admissionInteractionPolicy(catalog.runtimeCapabilities.interactionPolicy)) !==
    canonicalJson(admissionInteractionPolicy(browserCatalog.runtimeCapabilities.interactionPolicy))) {
  throw new Error("browser catalog interaction policy drifted from the native catalog");
}
const browserModules = new Map(browserCatalog.modules.map((module) => [module.id, module]));
for (const nativeModule of catalog.modules) {
  const browserModule = browserModules.get(nativeModule.id);
  if (!browserModule) throw new Error(`browser catalog omitted module ${nativeModule.id}`);
  const browserSchemes = new Map(browserModule.schemes.map((scheme) => [scheme.id, scheme]));
  for (const nativeScheme of nativeModule.schemes) {
    const browserScheme = browserSchemes.get(nativeScheme.id);
    if (!browserScheme) {
      throw new Error(`browser catalog omitted scheme ${nativeModule.id}/${nativeScheme.id}`);
    }
    const canonical = (scheme: typeof nativeScheme) => JSON.stringify({
      maturity: scheme.maturity,
      constructible: scheme.constructible,
      runtimeSelectable: scheme.runtimeSelectable,
      readinessState: scheme.readinessState,
      coordinateSystems: [...scheme.capabilities.coordinateSystems].sort(),
      staggerings: [...scheme.capabilities.staggerings].sort(),
      backends: [...scheme.capabilities.backends].sort(),
    });
    if (canonical(nativeScheme) !== canonical(browserScheme)) {
      throw new Error(
        `browser catalog capability drift for ${nativeModule.id}/${nativeScheme.id}: ` +
        `native=${canonical(nativeScheme)} browser=${canonical(browserScheme)}`,
      );
    }
  }
  if (browserSchemes.size !== nativeModule.schemes.length) {
    throw new Error(`browser catalog invented schemes for module ${nativeModule.id}`);
  }
}
if (browserModules.size !== catalog.modules.length) {
  throw new Error("browser catalog invented modules absent from the native catalog");
}
const stateFields = authoritativeRuntimeStateFields(catalog);
const batch = manifest.map((path) => {
  const absolutePath = resolve(path);
  const parsed = parseYaml(readFileSync(absolutePath, "utf8")) as unknown;
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`${absolutePath}: YAML root is not an object`);
  }
  return {
    configPath: absolutePath,
    request: buildConfigPreflightRequest(
      reconcileFormConfig(
        mergeConfig(defaultsForSchema(), parsed as ConfigObject),
      ),
      catalog,
      stateFields,
      sourceMode === "--exact-document"
        ? { yamlText: readFileSync(absolutePath, "utf8"), yamlPath: absolutePath }
        : undefined,
    ),
  };
});

writeFileSync(resolve(outputArgument), JSON.stringify(batch), "utf8");
process.stdout.write(
  `[studio-preflight-batch] PASS: ${batch.length} requests built from exact YAML/catalog\n`,
);
