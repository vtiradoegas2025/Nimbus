import { existsSync } from "node:fs";
import { dirname, resolve as resolvePath } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

/** Mirrors the "@/*" -> "./src/*" alias in tsconfig.app.json and vite.config.ts.
 *  Node has no notion of the alias, so the hook has to expand it before the
 *  default resolver ever sees the specifier. */
const SRC_DIR = fileURLToPath(new URL("../../ui/src/", import.meta.url));

/*
 * Module resolve hook for the `node --experimental-strip-types` test scripts.
 *
 * The project's src/ modules use extension-less relative imports (the Vite +
 * tsc `allowImportingTsExtensions` convention). Node's ESM resolver requires
 * explicit extensions, so loading a src module that has an extension-less
 * runtime import (e.g. diagnostics.ts -> "./configSchema") fails with
 * ERR_MODULE_NOT_FOUND. This hook appends `.ts` / `.tsx` to extension-less
 * relative specifiers when a matching sibling file exists, then defers to the
 * default resolver (the built-in type-stripping load hook handles the .ts URL).
 *
 * Used only by the test runner; production builds go through Vite, which
 * resolves these imports natively.
 */
const HAS_EXTENSION = /\.[cm]?[jt]sx?$/i;

function probe(base, context, nextResolve) {
  if (HAS_EXTENSION.test(base) && existsSync(base)) {
    return nextResolve(pathToFileURL(base).href, context);
  }
  for (const ext of [".ts", ".tsx"]) {
    if (existsSync(base + ext)) {
      return nextResolve(pathToFileURL(base + ext).href, context);
    }
  }
  return null;
}

/*
 * Vite's asset-URL queries. `import url from "./x.ts?worker&url"` asks Vite for
 * a built chunk's address; node instead resolves the query away and executes
 * the module, which for a worker means running its top-level `self` handlers on
 * the main thread. Redirected to a stub that explains itself.
 */
const VITE_ASSET_QUERY = /\?(worker|sharedworker|url|raw)(&|$)/;
const ASSET_URL_STUB = new URL("./vite-asset-url-stub.mjs", import.meta.url).href;

export async function resolve(specifier, context, nextResolve) {
  if (specifier === "gifenc") {
    // Node cannot discover this package's generated CommonJS named exports.
    return nextResolve("gifenc/dist/gifenc.esm.js", context);
  }
  if (VITE_ASSET_QUERY.test(specifier)) {
    return nextResolve(ASSET_URL_STUB, context);
  }
  if (specifier.startsWith("@/")) {
    const hit = probe(resolvePath(SRC_DIR, specifier.slice(2)), context, nextResolve);
    if (hit) return hit;
  }
  if (
    context.parentURL &&
    (specifier.startsWith("./") || specifier.startsWith("../")) &&
    !HAS_EXTENSION.test(specifier)
  ) {
    const parentDir = dirname(fileURLToPath(context.parentURL));
    const hit = probe(resolvePath(parentDir, specifier), context, nextResolve);
    if (hit) return hit;
  }
  return nextResolve(specifier, context);
}
