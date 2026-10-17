/*
 * Copy the subset of Pyodide the browser tests need into the harness's public
 * directory, so specs can boot a real CPython interpreter without the network.
 *
 * Where the pieces come from
 * --------------------------
 * The npm package carries the interpreter and the stdlib but NOT the package
 * index -- 13 MB of runtime and zero wheels. So the runtime files are copied
 * from node_modules, which guarantees they match the pinned version, and the
 * wheels are downloaded once from the CDN release of that same version, cached
 * under build/, and verified against the digests the lock file already
 * records. After the first run the whole thing is offline.
 *
 * Why a subset
 * ------------
 * The index is 310 packages. The tests boot with numpy and matplotlib, whose
 * transitive closure is a small fraction of that, and downloading the rest
 * would cost minutes for wheels nothing imports.
 *
 * Why the closure is computed and not listed
 * ------------------------------------------
 * matplotlib's dependencies are not stable across Pyodide releases, and a
 * hand-written list that goes stale fails as a ModuleNotFoundError inside the
 * WASM sandbox during a page load -- a long way from the list that caused it.
 * pyodide-lock.json already declares `depends` per package, so the closure is
 * derived from the same data the loader itself resolves against.
 *
 * Why the package list is parsed from the runtime
 * -----------------------------------------------
 * The set vendored here has to be the set the app preloads. Duplicating it
 * would let the two drift, and the failure mode of that drift is a test
 * environment quietly more capable, or less capable, than production. It is
 * read out of pyodideEngine.ts -- the interpreter half, which runs in the
 * worker and is where the preload list lives -- instead, and the parse failing
 * is a hard error rather than a fallback to a guess.
 *
 * Run: node ../tests/ui/vendor_pyodide.mjs
 */

import { cp, mkdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { createHash } from "node:crypto";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const uiRoot = fileURLToPath(new URL("../../ui/", import.meta.url));
const pyodideDir = join(uiRoot, "node_modules", "pyodide");
const outDir = join(uiRoot, "build", "test-public", "pyodide");
const runtimeSourcePath = join(uiRoot, "src", "lib", "platform", "pyodideEngine.ts");

/*
 * The loader and its payload. Every one of these is required: pyodide.mjs is
 * what the worker imports, pyodide.asm.* is the interpreter, python_stdlib.zip
 * is CPython's standard library, and the lock file is how loadPackage resolves
 * a name to a wheel. A missing entry here surfaces as a boot hang rather than
 * a clear error, so absence is fatal below rather than skipped.
 *
 * pyodide.mjs rather than pyodide.js: a worker has no document to append a
 * script tag to, so the classic build is unreachable from the thread the
 * interpreter now runs on.
 */
const CORE_FILES = [
  "pyodide.mjs",
  "pyodide.asm.js",
  "pyodide.asm.wasm",
  "python_stdlib.zip",
  "pyodide-lock.json",
];

/* Present in some releases and not others; copied when found. */
const OPTIONAL_CORE_FILES = ["pyodide.mjs.map", "pyodide-lock.json.map"];

/*
 * Vendored for the tests only, on top of what the app preloads.
 *
 * The install path cannot be exercised with a package that is already loaded,
 * so the specs need ones that are absent at boot and present on disk. micropip
 * ships with every Pyodide release, is not pulled in by numpy or matplotlib,
 * and is the same package the micropip fallback bootstraps -- so vendoring it
 * makes the install spec real rather than adding a fixture the app never uses.
 *
 * The other two exist for one claim: that a cell importing SEVERAL absent
 * packages resolves them in a single run. With only micropip absent that is
 * unfalsifiable, since one is indistinguishable from all. They are the two
 * smallest pure-Python wheels in the index that nothing else here depends on,
 * so the cost is a few tens of kilobytes on the CI cache.
 */
// xarray is also needed to exercise dataset()/load_dataset(): unlike explicit
// imports, those helpers import it internally, so their dependency path needs
// a real browser qualification rather than a source-shape double.
const TEST_ONLY_PACKAGES = ["micropip", "idna", "sortedcontainers", "xarray"];

function fail(message) {
  console.error(`vendor_pyodide: ${message}`);
  process.exit(1);
}

/** The packages the app preloads, read from the runtime so the two cannot drift. */
async function preloadedPackages() {
  const source = await readFile(runtimeSourcePath, "utf8");
  const match = /const PRELOADED_PACKAGES = \[([^\]]*)\]/.exec(source);
  if (!match) {
    fail(
      `could not find PRELOADED_PACKAGES in ${runtimeSourcePath}. If it was renamed, ` +
        "update this parse -- do not replace it with a hard-coded list, which is how " +
        "the test environment starts diverging from the app.",
    );
  }
  const names = [...match[1].matchAll(/"([^"]+)"/g)].map((entry) => entry[1]);
  if (names.length === 0) fail("PRELOADED_PACKAGES parsed as empty");
  return names;
}

/**
 * Every package name reachable from `roots` through the lock file's `depends`.
 *
 * Names in the lock file are canonical (lowercase, hyphenated) while an import
 * name may differ, so lookups go through the index's own key set rather than
 * assuming the two match.
 */
function resolveClosure(lock, roots) {
  const byName = new Map();
  for (const [key, entry] of Object.entries(lock.packages)) {
    byName.set(key.toLowerCase(), { key, entry });
  }
  const seen = new Set();
  const missing = [];
  const queue = [...roots];
  while (queue.length > 0) {
    const name = queue.pop();
    const canonical = name.toLowerCase();
    if (seen.has(canonical)) continue;
    const found = byName.get(canonical);
    if (!found) {
      missing.push(name);
      continue;
    }
    seen.add(canonical);
    for (const dependency of found.entry.depends ?? []) queue.push(dependency);
  }
  if (missing.length > 0) {
    fail(`packages absent from the Pyodide index: ${missing.join(", ")}`);
  }
  return [...seen].map((name) => byName.get(name));
}

function cacheDir(version) {
  // Under build/ with the other generated artifacts, and keyed by version so a
  // Pyodide bump cannot serve last release's wheels out of a stale cache.
  return join(uiRoot, "build", "pyodide-cache", version);
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

/**
 * Locate a package wheel, downloading it once if it is not on disk.
 *
 * The npm package carries the interpreter but not the package index -- 13 MB of
 * runtime, no wheels -- so the wheels have to come from the CDN. They are
 * cached under build/ and verified against the digest the lock file already
 * records, which is a stronger guarantee than the copy-from-node_modules path
 * had: a truncated download or a substituted file is rejected here rather than
 * failing later as an unreadable archive inside the WASM sandbox.
 */
async function wheelPath(entry, version) {
  const bundled = join(pyodideDir, entry.file_name);
  if (existsSync(bundled)) return { path: bundled, fetched: false };

  const cached = join(cacheDir(version), entry.file_name);
  if (existsSync(cached)) {
    const digest = sha256(await readFile(cached));
    if (!entry.sha256 || digest === entry.sha256) return { path: cached, fetched: false };
    // Corrupt or stale: fall through and replace it rather than serve it.
  }

  const url = `https://cdn.jsdelivr.net/pyodide/v${version}/full/${entry.file_name}`;
  let response;
  try {
    response = await fetch(url);
  } catch (error) {
    fail(`could not download ${entry.file_name}: ${error.message}`);
  }
  if (!response.ok) {
    fail(`could not download ${entry.file_name}: HTTP ${response.status} from ${url}`);
  }
  const bytes = Buffer.from(await response.arrayBuffer());
  if (entry.sha256) {
    const digest = sha256(bytes);
    if (digest !== entry.sha256) {
      fail(
        `checksum mismatch for ${entry.file_name}\n` +
          `  expected ${entry.sha256}\n  received ${digest}`,
      );
    }
  }
  await mkdir(cacheDir(version), { recursive: true });
  await writeFile(cached, bytes);
  return { path: cached, fetched: true };
}

async function main() {
  if (!existsSync(pyodideDir)) {
    fail(
      "node_modules/pyodide is missing. It is a devDependency pinned to the same " +
        "release the app loads from the CDN; run `npm install`.",
    );
  }

  const roots = await preloadedPackages();
  const lock = JSON.parse(await readFile(join(pyodideDir, "pyodide-lock.json"), "utf8"));

  /*
   * The vendored copy and the CDN URL must name the same release. Booting a
   * loader from one version against an index from another produces failures
   * deep inside the WASM runtime, so it is checked here where the message can
   * still say what is wrong.
   */
  const runtimeSource = await readFile(runtimeSourcePath, "utf8");
  const pinned = /const PYODIDE_VERSION = "([^"]+)"/.exec(runtimeSource)?.[1];
  if (!pinned) fail("could not read PYODIDE_VERSION from the runtime");
  if (lock.info?.version && lock.info.version !== pinned) {
    fail(
      `version mismatch: the app pins ${pinned} but node_modules/pyodide is ` +
        `${lock.info.version}. Update both together.`,
    );
  }

  await rm(outDir, { recursive: true, force: true });
  await mkdir(outDir, { recursive: true });

  for (const file of CORE_FILES) {
    const from = join(pyodideDir, file);
    if (!existsSync(from)) fail(`required runtime file missing from the package: ${file}`);
    await cp(from, join(outDir, file));
  }
  for (const file of OPTIONAL_CORE_FILES) {
    const from = join(pyodideDir, file);
    if (existsSync(from)) await cp(from, join(outDir, file));
  }

  const closure = resolveClosure(lock, [...roots, ...TEST_ONLY_PACKAGES]);
  let bytes = 0;
  let fetched = 0;
  for (const { entry } of closure) {
    const source = await wheelPath(entry, pinned);
    if (source.fetched) fetched += 1;
    await cp(source.path, join(outDir, entry.file_name));
    bytes += (await stat(source.path)).size;
  }
  if (fetched > 0) {
    console.log(`vendor_pyodide: downloaded ${fetched} wheel(s) into ${cacheDir(pinned)}`);
  }

  const names = closure.map(({ key }) => key).sort();
  console.log(
    `vendor_pyodide: ${pinned} -> build/test-public/pyodide\n` +
      `  app roots:  ${roots.join(", ")}\n` +
      `  test extra: ${TEST_ONLY_PACKAGES.join(", ")}\n` +
      `  closure:    ${closure.length} packages, ${(bytes / 1e6).toFixed(1)} MB\n` +
      `  packages:   ${names.join(", ")}`,
  );
}

await main();
