/*
 * Pyodide package-table drift checks.
 *
 * These tables are generated from one pinned Pyodide release. Nothing in the
 * type system ties them to that release, so the failure mode is silent: bump
 * PYODIDE_VERSION without regenerating and import aliases resolve against the
 * wrong index — only for the packages that moved, only at runtime, only for the
 * user who imported one.
 *
 * Run:
 *   node --experimental-strip-types --import ../tests/ui/register-ts-resolve.mjs \
 *     ../tests/ui/unit/platform/test_pyodide_packages.ts
 */

import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { pythonProviderResult } from "../../../../ui/src/lib/derived/pythonProviderResult.ts";

import {
  PYODIDE_CATALOG,
  PYODIDE_IMPORT_TO_PACKAGE,
  PYODIDE_LOCK_VERSION,
  PYODIDE_PACKAGE_NAMES,
} from "../../../../ui/src/lib/platform/pyodidePackages.ts";
import {
  detectMissingModule,
  disposePythonRuntimeClient,
  getPythonRuntimeClient,
  isBundledPyodideImport,
  packageNameForImport,
} from "../../../../ui/src/lib/platform/pyodideRuntime.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(`FAIL ${label}`);
}

const HERE = dirname(fileURLToPath(import.meta.url));
/*
 * The interpreter and its embedded Python live in the engine, which runs in the
 * worker; `pyodideRuntime` is now only the main-thread client. Everything
 * scanned below is Python or boot configuration, so the engine is the file that
 * holds it — while the three import-table lookups are still imported from the
 * client above, which is what proves the UI's import surface survived the move.
 */
const RUNTIME_SOURCE = resolve(HERE, "../../../../ui/src/lib/platform/pyodideEngine.ts");
const runtimeSource = readFileSync(RUNTIME_SOURCE, "utf8");
const CLIENT_SOURCE = resolve(HERE, "../../../../ui/src/lib/platform/pyodideRuntime.ts");
const clientSource = readFileSync(CLIENT_SOURCE, "utf8");
const WORKER_SOURCE = resolve(HERE, "../../../../ui/src/lib/platform/pyodideWorker.ts");
const workerSource = readFileSync(WORKER_SOURCE, "utf8");
const PROVIDER_SOURCE = resolve(HERE, "../../../../ui/src/lib/derived/pyodideDerivedProviderAdapter.ts");
const providerSource = readFileSync(PROVIDER_SOURCE, "utf8");

// ---------- 1. The tables describe the runtime that is actually loaded ----------

const pinnedVersion = /const PYODIDE_VERSION = "([\d.]+)"/.exec(runtimeSource)?.[1] ?? null;
check(pinnedVersion !== null, "the engine pins a PYODIDE_VERSION");
check(
  pinnedVersion === PYODIDE_LOCK_VERSION,
  `generated tables (${PYODIDE_LOCK_VERSION}) match the pinned runtime (${pinnedVersion})`,
);
check(
  /checkIntegrity: true/.test(runtimeSource),
  "automatic imports retain the lock-file wheel integrity check",
);

// Anti-vacuity: an empty table would satisfy every per-entry assertion below.
check(PYODIDE_PACKAGE_NAMES.size > 250, `package table is populated (${PYODIDE_PACKAGE_NAMES.size})`);
check(
  Object.keys(PYODIDE_IMPORT_TO_PACKAGE).length > 50,
  `alias table is populated (${Object.keys(PYODIDE_IMPORT_TO_PACKAGE).length})`,
);

// ---------- 2. Every alias points at a package that exists ----------

for (const [importName, packageName] of Object.entries(PYODIDE_IMPORT_TO_PACKAGE)) {
  check(
    PYODIDE_PACKAGE_NAMES.has(packageName),
    `alias '${importName}' resolves to a real package ('${packageName}')`,
  );
  // An identity entry is dead weight and hides a regenerate bug.
  check(importName !== packageName, `alias '${importName}' is not an identity mapping`);
}

// ---------- 3. Aliases the old hand-written table got wrong ----------

/*
 * The previous table had six entries against 69 real mismatches. It also spelled
 * PIL's package "pillow" when the index says "Pillow", and it missed both
 * case-only mappings entirely — `cartopy` -> `Cartopy` and `netCDF4` ->
 * `netcdf4` — so two packages that ship with Pyodide were unreachable. These
 * pin the cases that matter for a plotting surface.
 */
check(packageNameForImport("pylab") === "matplotlib", "'pylab' resolves to matplotlib");
check(packageNameForImport("mpl_toolkits") === "matplotlib", "'mpl_toolkits' resolves to matplotlib");
check(packageNameForImport("sklearn") === "scikit-learn", "'sklearn' resolves to scikit-learn");
check(packageNameForImport("PIL") === "Pillow", "'PIL' resolves to Pillow with the index's capitalisation");
check(packageNameForImport("netCDF4") === "netcdf4", "'netCDF4' resolves to netcdf4");
check(packageNameForImport("attr") === "attrs", "'attr' resolves to attrs");

// A name with no alias must fall through unchanged — that is the correct input
// for the micropip fallback, which takes the PyPI name.
check(packageNameForImport("xarray") === "xarray", "unaliased import falls through unchanged");
check(packageNameForImport("metpy") === "metpy", "package absent from the index falls through unchanged");

check(isBundledPyodideImport("xarray"), "xarray is recognised as bundled");
check(isBundledPyodideImport("pylab"), "an aliased import is recognised as bundled");
check(!isBundledPyodideImport("metpy"), "metpy is correctly reported as not bundled");

// ---------- 4. The catalog does not promise anything the index lacks ----------

for (const entry of PYODIDE_CATALOG) {
  if (entry.availability === "micropip") {
    // The whole point of this tier is that Pyodide does NOT carry it. If one
    // starts shipping, it should be promoted rather than left understated.
    check(
      !PYODIDE_PACKAGE_NAMES.has(entry.packageName),
      `'${entry.importName}' is listed as micropip-only and is genuinely absent from the index`,
    );
    continue;
  }
  check(
    PYODIDE_PACKAGE_NAMES.has(entry.packageName),
    `catalog entry '${entry.importName}' names a real package ('${entry.packageName}')`,
  );
  check(
    packageNameForImport(entry.importName) === entry.packageName,
    `catalog entry '${entry.importName}' agrees with the alias table`,
  );
  check(entry.purpose.trim().length > 0, `catalog entry '${entry.importName}' explains itself`);
}

const preloaded = PYODIDE_CATALOG.filter((entry) => entry.availability === "preloaded");
check(
  preloaded.some((entry) => entry.importName === "numpy")
    && preloaded.some((entry) => entry.importName === "matplotlib"),
  "numpy and matplotlib are both listed as preloaded",
);

// The catalog's preloaded tier must match what the runtime actually loads at
// boot, or the tab documents a package the user still has to install by hand.
const preloadList = /const PRELOADED_PACKAGES = \[([^\]]*)\]/.exec(runtimeSource)?.[1] ?? "";
const preloadNames = [...preloadList.matchAll(/"([^"]+)"/g)].map(([, name]) => name);
check(preloadNames.length > 0, "runtime declares a preload list");
check(
  preloaded.every((entry) => preloadNames.includes(entry.packageName)),
  `every catalog 'preloaded' entry is really preloaded (runtime loads ${preloadNames.join(", ")})`,
);
check(
  preloadNames.every((name) => preloaded.some((entry) => entry.packageName === name)),
  "every preloaded package is documented in the catalog",
);

// ---------- 5. matplotlib is usable without a canvas ----------

/*
 * Pyodide's matplotlib pulls in matplotlib-pyodide, which registers an
 * interactive backend expecting a live DOM target. Studio renders by calling
 * savefig into a buffer, so the backend must be forced to AGG before pyplot is
 * first imported — otherwise figure capture depends on a canvas that is never
 * mounted.
 */
check(
  /matplotlib\.use\("AGG"\)/.test(runtimeSource),
  "runtime forces the non-interactive AGG backend",
);
// Anchor on the exact call that runs user code, not the first `runPythonAsync`
// in the file — micropipInstall uses one too, and matching that would compare
// against the wrong position.
const USER_CELL_CALL = "await pyodide.runPythonAsync(source)";
const userCellAt = runtimeSource.indexOf(USER_CELL_CALL);
check(userCellAt > 0, "the user-cell execution call is locatable");
check(
  runtimeSource.indexOf("pyodide.runPython(MATPLOTLIB_BACKEND_SETUP)") < userCellAt,
  "the backend is selected during boot, before any user cell runs",
);

// Idiomatic matplotlib assigns nothing; the newest live figure is the result.
check(
  /get_fignums\(\)/.test(runtimeSource),
  "result extraction falls back to an open figure when 'result' is unset",
);
check(
  !/__nimbus_user_result__ = _plt\.gcf\(\)/.test(runtimeSource),
  "the fallback avoids gcf(), which would create a blank figure",
);
/*
 * A figure left by a previous cell must not be credited to this one, so the
 * reset has to happen in the setup block that precedes the user's cell.
 *
 * Anchored on the stash of the real streams rather than on the assignment that
 * replaces them: the assignment's shape is an implementation detail that has
 * already changed once (a plain StringIO became a tee that also forwards each
 * chunk to the UI), while the stash is what the restore in `finally` depends
 * on and cannot go away while capture works at all.
 */
const redirectAt = runtimeSource.indexOf("_orig_stdout, _orig_stderr = sys.stdout, sys.stderr");
check(redirectAt > 0 && redirectAt < userCellAt, "the pre-exec setup block precedes the user cell");
const setupBlock = runtimeSource.slice(redirectAt, userCellAt);
check(
  /close\("all"\)/.test(setupBlock),
  "stale figures are discarded before the user's cell runs",
);
// Checked separately from the anchor so that redirecting only one stream --
// which would silently send the other to the console the user cannot see --
// fails here rather than passing on the strength of the stash alone.
check(
  /sys\.stdout\s*=/.test(setupBlock) && /sys\.stderr\s*=/.test(setupBlock),
  "both streams are captured before the user's cell runs",
);

/*
 * Live output.
 *
 * The tee has to write to the buffer as well as the sink: the buffer is what
 * the run's final stdout/stderr are read from, so a tee that only forwarded
 * would show output while the cell ran and then replace it with nothing.
 */
check(
  /self\._buffer\.write\(text\)/.test(runtimeSource),
  "the streaming tee still writes to the captured buffer",
);
check(
  /self\._sink\(self\._name, text\)/.test(runtimeSource),
  "the streaming tee forwards each chunk to the UI",
);
/*
 * The sink closes over React state. Leaving it on the interpreter's globals
 * would hold that run's setState -- and everything it references -- for the
 * rest of the session, so the release is pinned here.
 */
const runCellAt = runtimeSource.indexOf("export async function runPythonCell");
check(
  /globals\.set\("_nimbus_stream_sink", null\)/.test(runtimeSource.slice(runCellAt)),
  "the stream sink is released when the run ends",
);

// ---------- 6. Missing-module detection feeds the alias table ----------

check(
  detectMissingModule("ModuleNotFoundError: No module named 'xarray'") === "xarray",
  "a plain missing module is detected",
);
check(
  detectMissingModule("ModuleNotFoundError: No module named 'scipy.signal'") === "scipy",
  "a submodule is reduced to its top-level package",
);
check(
  detectMissingModule("ModuleNotFoundError: No module named 'mpl_toolkits.mplot3d'") === "mpl_toolkits",
  "an aliased submodule reduces to the import name the alias table keys on",
);
check(
  packageNameForImport(
    detectMissingModule("ModuleNotFoundError: No module named 'mpl_toolkits.mplot3d'") ?? "",
  ) === "matplotlib",
  "detection and alias resolution compose into a loadable package",
);
check(detectMissingModule("ValueError: nope") === null, "unrelated errors are not misread");

// ---------- 7. The xarray helper is bound with the frame ----------

check(
  /def dataset\(fields=None, valid_only=True\)/.test(runtimeSource),
  "bindContext defines a dataset() helper",
);
check(
  /_bound_fields/.test(runtimeSource) && /_field_units/.test(runtimeSource),
  "dataset() is given the bound field names and their units",
);
check(
  /import xarray as xr/.test(runtimeSource),
  "dataset() imports xarray bare so ModuleNotFoundError stays detectable",
);
check(
  /z_m\.size == nz/.test(runtimeSource),
  "dataset() only uses physical height when a separable z coordinate exists",
);
check(
  /array\[slices\]/.test(runtimeSource) && /array\[slices\[:2\]\]/.test(runtimeSource),
  "dataset() excludes simulator storage halos unless explicitly requested",
);

// ---------- 8. The tab's own documentation matches the catalog ----------

/*
 * The starter cell names the available packages in prose. Prose and catalog
 * drift apart silently — a package added to one is simply absent from the
 * other, and the user reads whichever is stale. Read the component as text
 * because it is TSX and cannot be imported under strip-only type stripping.
 */
const PYTHON_PLOT_SOURCE = resolve(HERE, "../../../../ui/src/components/plots/PythonPlot.tsx");
const pythonPlotSource = readFileSync(PYTHON_PLOT_SOURCE, "utf8");
const starterCode = /const STARTER_CODE = `([\s\S]*?)`;/.exec(pythonPlotSource)?.[1] ?? "";
check(starterCode.length > 0, "the starter cell is locatable");

for (const entry of PYODIDE_CATALOG) {
  if (entry.availability !== "bundled") continue;
  check(
    starterCode.includes(entry.importName),
    `starter cell mentions bundled package '${entry.importName}'`,
  );
}
check(
  starterCode.includes("dataset()"),
  "starter cell documents the dataset() helper the catalog points at",
);
// The catalog claims matplotlib renders without assigning `result`; the starter
// cell must say so too, since that is the behaviour users would otherwise not
// discover.
check(
  /current figure is captured/.test(starterCode),
  "starter cell documents automatic figure capture",
);

// The panel that renders the catalog has to actually consume it, or the catalog
// is documentation that ships to nobody.
check(
  /PYODIDE_CATALOG/.test(pythonPlotSource),
  "the Python tab renders the catalog rather than exporting it unused",
);
check(
  /isBundledPyodideImport/.test(pythonPlotSource),
  "the missing-module banner distinguishes bundled from micropip-only packages",
);

// ---------- 9. Multi-frame access ----------

/*
 * Only the on-screen frame is injected as globals, because injecting a run
 * would copy every field of every step into the WASM heap. `load_field` exposes
 * the reader instead. Two properties matter: it must not exist when no reader
 * was supplied (a helper that always raises still autocompletes and reads as
 * supported), and it must not leak the temporary globals it uses to cross the
 * FFI, or a long session retains every frame it ever touched.
 */
// Signatures themselves are pinned in section 14, alongside the source
// argument they gained; here we only care that the helpers exist at all.
check(/async def load_field\(/.test(runtimeSource), "a frame loader is defined for Python");
check(/async def load_series\(/.test(runtimeSource), "a frame-stacking helper is defined");
check(
  /globals\(\)\.pop\('load_field', None\)/.test(runtimeSource),
  "load_field is removed when the widget supplies no reader",
);
check(
  /globals\(\)\.pop\(key \+ "_buffer", None\)/.test(runtimeSource)
    && /globals\(\)\.pop\(key \+ "_shape", None\)/.test(runtimeSource),
  "the loader clears the temporaries it parks in the namespace",
);
check(
  /if not \(0 <= target < limit\)/.test(runtimeSource),
  "an out-of-range frame is rejected before a read is attempted",
);
// The array must cross by the same route as the injected fields; a bespoke
// conversion is where a silent dtype or shape bug would live.
check(
  /\$\{key\} = np\.asarray\(\$\{key\}_buffer, dtype=np\.float32\)\.reshape/.test(runtimeSource),
  "loaded frames use the same proven array conversion as bound fields",
);
check(
  /frame_index/.test(runtimeSource) && /frame_count/.test(runtimeSource)
    && /frame_times_s/.test(runtimeSource),
  "frame position, count and times are all bound",
);
check(
  starterCode.includes("load_field") && starterCode.includes("load_series"),
  "the starter cell documents multi-frame access",
);

// ---------- 10. External data files ----------

check(
  /files\["cm1out\.nc"\]/.test(starterCode) || starterCode.includes('files["'),
  "the starter cell documents the files dict",
);
check(
  /bindStagedFiles/.test(pythonPlotSource),
  "the widget binds staged files into the interpreter",
);
// Rebinding on every run is what makes a released file disappear rather than
// leaving a path that no longer resolves.
const runBindAt = pythonPlotSource.indexOf("await bindStagedFiles(stagedFiles,");
check(
  runBindAt > 0
    && pythonPlotSource.indexOf("runtime.bindStagedFilesInPython.bind(runtime)", runBindAt) > runBindAt,
  "staged files are rebound into their pane-scoped runtime as part of the run path",
);

// ---------- 11. The display protocol ----------

/*
 * A cell emits a LIST of outputs. The properties that matter: explicit
 * display() wins; the old single-value behaviour still works as a fallback so
 * existing snippets keep rendering; and outputs survive an exception, because a
 * cell that drew three figures and then raised should still show the figures.
 */
check(/def display\(\*values, label=None\)/.test(runtimeSource), "display() accepts several values");
check(/_nimbus_outputs\.append/.test(runtimeSource), "display() accumulates rather than replacing");
check(
  /if not _nimbus_outputs:/.test(runtimeSource),
  "the single-value fallback applies only when nothing was displayed",
);
check(
  /_fallback = globals\(\)\.get\("result", None\)/.test(runtimeSource),
  "an assigned `result` is still rendered",
);
check(
  /if _nimbus_result_assigned else None/.test(runtimeSource),
  "a print-only cell cannot redisplay an earlier cell's stale result",
);
check(
  /get_fignums\(\)/.test(runtimeSource) && !/_plt\.gcf\(\)/.test(runtimeSource),
  "the implicit-figure fallback still avoids creating a blank figure",
);
const failurePath = runtimeSource.indexOf("stderr: stderr || (exception");
check(
  failurePath > 0
    && runtimeSource.slice(failurePath, failurePath + 200).includes("collectOutputs"),
  "outputs are returned even when the cell raised",
);
check(
  /_nimbus_outputs = \[\]/.test(runtimeSource.slice(0, userCellAt)),
  "the output list is cleared before each run",
);
check(
  /_NIMBUS_MAX_OUTPUT_ITEM_BYTES/.test(runtimeSource)
    && /_NIMBUS_MAX_OUTPUT_TOTAL_BYTES/.test(runtimeSource)
    && /_NIMBUS_MAX_HTML_OUTPUT_BYTES/.test(runtimeSource)
    && /_NIMBUS_MAX_ANIMATION_OUTPUT_BYTES/.test(runtimeSource),
  "arrays, HTML, animations, and the whole cell have explicit transfer budgets",
);
check(
  /entry\["bytes"\] = int\(value\.nbytes\)/.test(runtimeSource)
    && /next_total = _nimbus_output_bytes/.test(runtimeSource),
  "output budgets use actual encoded/array bytes before values cross the worker boundary",
);
check(
  /declaredBytes < 0 \|\| declaredBytes > itemLimit/.test(runtimeSource)
    && /transferredBytes \+ declaredBytes > MAX_OUTPUT_TOTAL_BYTES/.test(runtimeSource),
  "the UI defensively rechecks every declared output budget before copying it from wasm",
);
check(
  /class _NimbusCappedText/.test(runtimeSource)
    && /_NIMBUS_MAX_CAPTURED_STREAM_CHARS/.test(runtimeSource)
    && /final stream capture truncated/.test(runtimeSource),
  "final stdout and stderr capture is bounded inside the interpreter",
);
check(
  runtimeSource.includes('value += "\\\\n[Nimbus: final stream capture truncated at 2 MiB]\\\\n"'),
  "the stream-limit notice remains a valid Python string after template evaluation",
);
check(
  /retainRunResult/.test(pythonPlotSource)
    && /MAX_OUTPUT_TOTAL_BYTES - retainedRunBytes/.test(pythonPlotSource)
    && /retentionOrder/.test(pythonPlotSource),
  "multi-cell and retained notebook results share one bounded memory budget",
);
check(
  /actualStructuredBytes !== declaredBytes/.test(runtimeSource)
    && /Displayed array shape or byte metadata did not match its buffer/.test(runtimeSource),
  "encoded output bytes and ndarray shapes are revalidated before UI retention",
);
check(
  /function responseTransferables/.test(workerSource)
    && /WORKER_OPS\.readProducedFile/.test(workerSource)
    && /kind\?: unknown; data\?: unknown/.test(workerSource),
  "worker-owned produced files and ndarray results use transfer lists",
);
check(
  /Deliberately clone: the returned field may alias the shared frame cache/.test(clientSource),
  "reverse field loads preserve cached input-buffer ownership",
);
check(
  /class WorkerRuntimeClient/.test(clientSource)
    && /getPythonRuntimeClient/.test(clientSource)
    && /runtimeKey/.test(clientSource),
  "each pane can own an isolated worker and runtime-branded handle",
);
check(
  /throwIfCancelled\(signal\)/.test(providerSource)
    && /providerRuntime\.restartPython\(\)/.test(providerSource)
    && /PROVIDER_STOP_GRACE_MS/.test(providerSource),
  "derived-provider cancellation is rechecked and has a forceful isolated-runtime fallback",
);
const transferredProviderArray = Float32Array.of(1.25, -2.5);
const transferredProviderResult = pythonProviderResult({
  raised: false, stdout: "", stderr: "", produced: [],
  outputs: [{ kind: "ndarray", label: null, data: transferredProviderArray, shape: [2], dtype: "<f4" }],
});
check(
  /pythonProviderResult\(outcome\)/.test(providerSource)
    && /values: array\.values/.test(providerSource)
    && transferredProviderResult.values === transferredProviderArray.buffer,
  "derived providers reuse their transferred result instead of copying it again",
);
check(
  /def static_members\(target\)/.test(runtimeSource)
    && /_inspect\.getattr_static/.test(runtimeSource)
    && !/names = dir\(target\)/.test(runtimeSource),
  "completion enumerates attributes without invoking user-defined getters or __dir__",
);
check(
  /if isinstance\(value, np\.ndarray\)/.test(runtimeSource)
    && !/shape = getattr\(value, "shape"/.test(runtimeSource),
  "namespace details do not invoke an arbitrary object's shape property",
);

// Execute the exact embedded completion helper with CPython as a syntax and
// side-effect regression. It uses only stdlib semantics shared by Pyodide.
const completionStart = runtimeSource.indexOf("def _nimbus_complete(path, partial):");
const completionEnd = runtimeSource.indexOf("\n\n# Names this module injects", completionStart);
const completionHelper = runtimeSource.slice(completionStart, completionEnd);
const completionProbe = `
import json as _json
_NIMBUS_MISSING = object()
_NIMBUS_MAX_COMPLETIONS = 200
def _nimbus_kind(value):
    return type(value).__name__
def _nimbus_detail(value):
    return None
${completionHelper}
hits = 0
class Trap:
    @property
    def danger(self):
        global hits
        hits += 1
        return 1
    def __dir__(self):
        global hits
        hits += 100
        return ["danger"]
trap = Trap()
print(_nimbus_complete(["trap"], "danger"))
print(_nimbus_complete([], "wh"))
print(hits)
`;
const completionRun = spawnSync("python3", ["-c", completionProbe], { encoding: "utf8" });
check(completionRun.status === 0, "the embedded completion helper executes successfully");
check(
  completionRun.stdout.trim().split("\n").at(-1) === "0",
  "completion executes neither a property nor a custom __dir__",
);
check(
  completionRun.stdout.includes('"while"'),
  "root completion executes the keyword path without undefined state",
);

// Exercise the client transport with a deterministic Worker double. This
// verifies isolation and transfer ownership rather than only scanning source.
type FakeListener = (event: { data: unknown; message?: string }) => void;
class FakeWorker {
  static instances: FakeWorker[] = [];
  readonly calls: Array<{ message: any; transfer: Transferable[] }> = [];
  readonly listeners = new Map<string, FakeListener>();
  readonly url: string;
  readonly options: { name?: string };
  terminated = false;
  constructor(url: string, options: { name?: string }) {
    this.url = url;
    this.options = options;
    FakeWorker.instances.push(this);
  }
  addEventListener(type: string, listener: FakeListener): void { this.listeners.set(type, listener); }
  postMessage(message: any, transfer: Transferable[] = []): void {
    this.calls.push({ message, transfer });
    if (typeof structuredClone === "function") structuredClone(message, { transfer });
    queueMicrotask(() => this.listeners.get("message")?.({
      data: { kind: "response", id: message.id, ok: true, value: null },
    }));
  }
  terminate(): void { this.terminated = true; }
}
const originalWorker = (globalThis as any).Worker;
(globalThis as any).Worker = FakeWorker;
const clientA = getPythonRuntimeClient("test:runtime-a");
const clientAAgain = getPythonRuntimeClient("test:runtime-a");
const clientB = getPythonRuntimeClient("test:runtime-b");
check(clientA === clientAAgain, "a run key receives one stable runtime client");
const [handleA, handleB] = await Promise.all([clientA.getPyodide(), clientB.getPyodide()]);
check(
  FakeWorker.instances.length === 2 && handleA.runtimeKey !== handleB.runtimeKey,
  "different run keys create isolated workers and branded handles",
);
check(
  FakeWorker.instances[0]?.url !== FakeWorker.instances[1]?.url
    && FakeWorker.instances.every((worker) => worker.url.includes("runtime=")),
  "pane workers receive distinct module URLs so WebKit starts both runtimes",
);
let foreignHandleRejected = false;
try {
  await clientB.bindContext(handleA, { fields: [], grid: null, simulationTimeS: null });
} catch {
  foreignHandleRejected = true;
}
check(foreignHandleRejected, "a client rejects another runtime's handle before sending work");
check(
  clientA.tryAcquireExecution() && !clientA.tryAcquireExecution(),
  "execution admission is atomic for rapid duplicate runs",
);
clientA.restartPython();
check(
  !clientA.tryAcquireExecution(),
  "replacing a worker does not admit a new run before the stopped run unwinds",
);
clientA.releaseExecution();
const stagedPayload = new Uint8Array([1, 2, 3, 4]);
await clientA.writeStagedFile("/nimbus", "/nimbus/a.npy", stagedPayload);
check(stagedPayload.byteLength === 0, "owned staged bytes transfer instead of cloning");
check(
  FakeWorker.instances.some((worker) => worker.calls.some((call) => call.transfer.length === 1)),
  "the staged-file RPC carries an explicit transfer list",
);
disposePythonRuntimeClient("test:runtime-a");
disposePythonRuntimeClient("test:runtime-b");

// All-busy overflow is allowed temporarily so active scientific work is never
// killed. The first client to become idle must retry LRU eviction, otherwise a
// long project/pane session retains the overflow forever.
const overflowClients = Array.from({ length: 12 }, (_, index) =>
  getPythonRuntimeClient(`test:overflow-${index}`)
);
await overflowClients[0].getPyodide();
const oldestOverflowWorker = FakeWorker.instances.at(-1)!;
check(
  overflowClients.every((client) => client.tryAcquireExecution()),
  "scoped runtimes can all be busy without being terminated",
);
const newestOverflowClient = getPythonRuntimeClient("test:overflow-12");
check(!oldestOverflowWorker.terminated, "creating an overflow runtime preserves busy clients");
overflowClients[0].releaseExecution();
check(
  oldestOverflowWorker.terminated
    && getPythonRuntimeClient("test:overflow-12") === newestOverflowClient,
  "an idle transition retries LRU eviction and restores the scoped-client bound",
);
for (let index = 0; index <= 12; index += 1) disposePythonRuntimeClient(`test:overflow-${index}`);
(globalThis as any).Worker = originalWorker;
check(
  /MAX_PRODUCED_FILE_BYTES/.test(runtimeSource)
    && /Produced-file export path is outside/.test(runtimeSource)
    && /FS\.isFile/.test(runtimeSource),
  "produced-file export is regular-file, path, stability, and transfer-budget checked",
);

// Capture order is load-bearing: several of these types satisfy more than one
// test, so a looser ordering would render an animation as a bare repr.
const animationAt = runtimeSource.indexOf('isinstance(value, NimbusAnimation)');
const jshtmlAt = runtimeSource.indexOf('hasattr(value, "to_jshtml")');
const savefigAt = runtimeSource.indexOf('hasattr(value, "savefig")');
const reprHtmlAt = runtimeSource.indexOf('hasattr(value, "_repr_html_")');
check(animationAt > 0 && jshtmlAt > animationAt, "NimbusAnimation is matched before to_jshtml");
check(jshtmlAt < savefigAt, "an animation is matched before a plain figure");
check(savefigAt < reprHtmlAt, "a figure is matched before a generic HTML repr");

// xarray and pandas both ship _repr_html_; rendering it is the difference
// between a readable table and a wall of repr() text.
check(reprHtmlAt > 0, "objects that can render themselves as HTML do so");

// ---------- 12. Animation ----------

check(/def animate\(frames, fps=10\)/.test(runtimeSource), "animate() builds a playable sequence");
check(
  /_NIMBUS_MAX_ANIMATION_FRAMES/.test(runtimeSource),
  "the frame count is bounded",
);
check(
  /_plt\.close\(figure\)/.test(runtimeSource) && /_plt\.close\(item\)/.test(runtimeSource),
  "animate() closes each figure it renders rather than leaking the whole sequence",
);
check(
  starterCode.includes("animate(") || starterCode.includes("display("),
  "the starter cell mentions the display/animation API",
);

// ---------- 13. Export ----------

check(/def save_path\(name\)/.test(runtimeSource), "save_path() gives user code an export target");
check(
  /takes a bare file name, not a path/.test(runtimeSource),
  "save_path() rejects a path, which would write outside the export directory",
);
check(
  runtimeSource.includes('name in (".", "..")')
    && runtimeSource.includes('or "\\\\\\\\" in name'),
  "save_path() rejects parent-directory aliases and cross-platform separators",
);
check(
  /_shutil\.rmtree/.test(runtimeSource.slice(0, userCellAt)),
  "the export directory is cleared before each run so the panel shows this run's files",
);
/*
 * ...but not for a fragment that starts mid-document. Running one cell of a
 * script whose earlier cell saved a dataset would otherwise delete that file
 * as a side effect of drawing from it, which is the export path losing user
 * data rather than merely displaying it oddly.
 */
check(
  /resetProducedFiles === false/.test(runtimeSource.slice(0, userCellAt)),
  "the clear is skipped for a fragment that does not start at the top",
);
check(
  /_os\.makedirs/.test(runtimeSource.slice(0, userCellAt)),
  "the export directory is created either way, so save_path() always resolves",
);
check(
  /export function readProducedFile/.test(runtimeSource),
  "produced files can be read back for export",
);

// HTML from Python is rendered in a sandboxed frame, never injected into the
// app's DOM: repr payloads carry their own CSS and the animation player carries
// script, and neither should reach the surrounding document.
check(
  /sandbox=\{output\.scripts \? "allow-scripts" : ""\}/.test(pythonPlotSource),
  "Python HTML renders in a sandboxed iframe, with scripts only where required",
);
// Matches the JSX prop specifically; a prose mention of the anti-pattern in a
// comment is not a use of it.
check(
  !/dangerouslySetInnerHTML=\{/.test(pythonPlotSource),
  "Python HTML is never injected directly into the app DOM",
);

// ---------- 14. Multiple datasets ----------

/*
 * Studio can hold a comparison dataset alongside the primary, already loaded
 * and frame-aligned, but the Python tab could previously only ever see the
 * primary. Every loader now takes source=, and the bound `datasets` dict is
 * what makes the set discoverable rather than guessable.
 */
check(
  // Open-ended on trailing parameters: stride= and box= were added so a field
  // can be reduced at the reader, which is what makes any domain size workable.
  /async def load_field\(name, frame=None, source="primary"[,)]/.test(runtimeSource),
  "load_field takes a source",
);
check(
  /async def load_series\(name, frames=None, source="primary", stride=1, box=None,\s+valid_only=True, terrain_remap=False\)/.test(runtimeSource),
  "load_series takes a source",
);
check(
  /async def load_dataset\(\s*fields, frame=None, source="primary", stride=1, box=None, valid_only=True,\s+terrain_remap=False/.test(runtimeSource),
  "an xarray view of another source is available",
);
check(
  /terrain_remap=True changes stride to physical-volume block reduction/.test(runtimeSource)
    && /def field_metadata\(/.test(runtimeSource)
    && /coordinate_resampling/.test(runtimeSource)
    && starterCode.includes("terrain_remap=terrain_following")
    && starterCode.includes('field_metadata("w")'),
  "terrain-aware Python loads disclose conservative remap coordinates and provenance",
);
check(/^datasets = \{$/m.test(runtimeSource), "the open datasets are bound for discovery");
check(
  /unknown source '\{source\}'; open datasets/.test(runtimeSource),
  "an unknown source names what is actually open",
);

/*
 * The comparison run is frequently a different length. Validating a frame
 * index against the primary's count would accept one that does not exist in
 * the source being read, so the bound is taken per-source.
 */
check(
  /limit = datasets\[source\]\["frames"\]/.test(runtimeSource),
  "the frame bound is read from the requested source",
);
check(
  /outside 0\.\.\{limit - 1\} for source/.test(runtimeSource),
  "an out-of-range frame reports the source it was checked against",
);
check(
  !/0 <= target < frame_count/.test(runtimeSource),
  "the primary's frame count is no longer used to bound every source",
);
check(
  /requested fields do not share the same spatial coordinates/.test(runtimeSource),
  "load_dataset rejects incompatible spatial coordinates rather than silently aligning them",
);
check(
  /_NIMBUS_MAX_DATASET_BYTES/.test(runtimeSource)
    && /next_bytes = retained_bytes \+ int\(array\.nbytes\)/.test(runtimeSource)
    && /request fewer fields, a tighter box, or a larger stride/.test(runtimeSource),
  "load_dataset bounds the combined retained field payload",
);
check(
  /usesDatasetHelper/.test(runtimeSource)
    && /stripPythonNonCode\(source\)/.test(runtimeSource)
    && /`\$\{source\}\\nimport xarray`/.test(runtimeSource),
  "dataset helpers preload their implicit xarray dependency without scanning comments or strings",
);
check(
  /_loaded_field_metadata/.test(runtimeSource)
    && /built\.attrs\["step_index"\]/.test(runtimeSource)
    && /built\.attrs\["simulation_time_s"\]/.test(runtimeSource),
  "on-demand xarray datasets preserve source, step, time, units, and coordinates",
);
check(
  /_NIMBUS_MAX_LOADED_FIELD_METADATA = 256/.test(runtimeSource)
    && /_loaded_field_metadata\.pop\(next\(iter\(_loaded_field_metadata\)\)\)/.test(runtimeSource),
  "on-demand coordinate metadata is bounded across long multi-frame sessions",
);
check(
  /attrs\["nimbus_sampling"\]/.test(runtimeSource)
    && /attrs\["nimbus_source_shape"\]/.test(runtimeSource)
    && /attrs\["nimbus_source_format"\]/.test(runtimeSource),
  "xarray variables retain the fidelity and serving-format provenance of each field",
);
check(
  /attrs\["nimbus_compression"\]/.test(runtimeSource)
    && /compression: step\.fieldCompression\?\.\[fieldName\] \?\? null/.test(pythonPlotSource),
  "Python and xarray expose the manifest-declared per-field compression contract",
);

// The widget must route a load to the requested dataset, not just accept the
// argument and read the primary anyway.
check(
  /sourceHandles\.find\(\(entry\) => entry\.role === source\)/.test(pythonPlotSource),
  "the widget resolves the requested source before reading",
);
check(
  /No open dataset for source/.test(pythonPlotSource),
  "an unresolvable source fails loudly rather than silently reading the primary",
);
// Mounted without the dataset list (harness, embed) the tab must still work.
check(
  /datasets && datasets\.length > 0/.test(pythonPlotSource),
  "a missing dataset list falls back to a synthetic primary handle",
);
check(
  starterCode.includes('source="comparison"'),
  "the starter cell shows how to read the other dataset",
);

console.log(`\npyodide package tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length) {
  console.log("\nFailures:");
  for (const failure of failures) console.log(`  ${failure}`);
  process.exit(1);
}
