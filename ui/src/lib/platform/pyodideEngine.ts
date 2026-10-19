import { pythonFieldCoordinates } from "@/lib/plot/pythonFieldCoordinates";
import { decodePythonNumericArray, pythonDtypeBytes, type PythonNumericArray } from "./pythonNumericArray";
/*
 * The interpreter itself. RUNS INSIDE A WEB WORKER — never on the UI thread.
 * Nothing here may touch window/document/React. Only importer is pyodideWorker.
 * Shares one Python namespace across cells (Jupyter UX); injects fields as numpy
 * arrays and grid metadata as globals via np.asarray(...).reshape(...).
 */

import type { CompletionCandidate } from "@/lib/studio/pythonCompletion";
import { stripPythonNonCode } from "@/lib/studio/pythonFieldRefs";
import { PYODIDE_LOCK_VERSION } from "./pyodidePackages";
import { isBundledPyodideImport, packageNameForImport } from "./pyodideImports";
import {
  MAX_ANIMATION_FRAMES,
  MAX_ANIMATION_OUTPUT_BYTES,
  MAX_CAPTURED_STREAM_CHARS,
  MAX_COMPLETIONS,
  MAX_DATASET_BYTES,
  MAX_HTML_OUTPUT_BYTES,
  MAX_OUTPUT_ITEM_BYTES,
  MAX_OUTPUT_TOTAL_BYTES,
  MAX_OUTPUTS_PER_RUN,
  MAX_PRODUCED_FILE_BYTES,
  MAX_PRODUCED_FILES,
  MAX_SERIES_BYTES,
  PYODIDE_OUTPUT_DIR,
  type InjectionContext,
  type NamespaceEntry,
  type ProducedFile,
  type PyodideStatusListener,
  type PythonOutput,
  type RunOptions,
  type RunResult,
} from "./pyodideProtocol";

/* eslint-disable @typescript-eslint/no-explicit-any */

// Pinned CDN release. Bump only after verifying API compatibility.
const PYODIDE_VERSION = "0.26.4";
const PYODIDE_CDN_BASE = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

/**
 * CDN in the shipped app; VITE_PYODIDE_BASE_URL overrides it for offline tests.
 * Normalises trailing slash so derived URLs don't land in the parent directory.
 */
function configuredPyodideBase(): string | null {
  try {
    // Vite substitutes this at build time; under plain node it throws (caught below).
    const configured = import.meta.env.VITE_PYODIDE_BASE_URL;
    if (typeof configured !== "string" || configured.length === 0) return null;
    return configured.endsWith("/") ? configured : `${configured}/`;
  } catch {
    return null;
  }
}

const PYODIDE_BASE_URL = configuredPyodideBase() ?? PYODIDE_CDN_BASE;

// numpy + matplotlib preloaded at boot. AGG backend selected before pyplot is
// first imported — savefig must not try to render an interactive canvas.
const PRELOADED_PACKAGES = ["numpy", "matplotlib"] as const;

const MATPLOTLIB_BACKEND_SETUP = `
import matplotlib
matplotlib.use("AGG")
`;

interface PyodideHandle {
  pyodide: any;
  /** Shared byte Pyodide polls on every bytecode instruction for KeyboardInterrupt. */
  interruptBuffer: Uint8Array;
}

let runtimeHandle: PyodideHandle | null = null;
let runtimeLoading: Promise<PyodideHandle> | null = null;
// Set while a cell runs; out-of-band callers (completion) must not interleave.
let cellInFlight = false;

/** Lazily boot Pyodide; concurrent callers share the same promise. */
export async function getPyodide(
  onStatus?: PyodideStatusListener,
  interruptBuffer?: Uint8Array,
): Promise<PyodideHandle> {
  if (runtimeHandle) {
    onStatus?.({ kind: "ready" });
    return runtimeHandle;
  }
  if (runtimeLoading) return runtimeLoading;

  runtimeLoading = (async () => {
    try {
      // Package tables are version-specific; mismatched runtime resolves aliases silently wrong.
      if (PYODIDE_LOCK_VERSION !== PYODIDE_VERSION) {
        throw new Error(
          `Pyodide package tables were generated from ${PYODIDE_LOCK_VERSION} but the runtime is `
          + `pinned to ${PYODIDE_VERSION}. Regenerate src/lib/platform/pyodidePackages.ts.`,
        );
      }
      onStatus?.({ kind: "loading-runtime" });
      const loadPyodide = await loadPyodideModule();
      // indexURL must match the loader's base: wasm, stdlib, and wheels resolve relative to it.
      const pyodide = await loadPyodide({ indexURL: PYODIDE_BASE_URL });

      onStatus?.({ kind: "loading-package", name: PRELOADED_PACKAGES.join(", ") });
      await pyodide.loadPackage([...PRELOADED_PACKAGES]);
      pyodide.runPython(MATPLOTLIB_BACKEND_SETUP);
      // numpy must exist before the display helpers, which reference it.
      pyodide.runPython("import numpy as np\n_nimbus_outputs = []\n_nimbus_output_sequence = 0");
      pyodide.runPython(DISPLAY_HELPER);

      // Interrupt sentinel: main thread writes this byte; Pyodide raises KeyboardInterrupt.
      // Falls back to a plain ArrayBuffer when COOP+COEP is absent (Stop is then disabled).
      const sentinel = interruptBuffer ?? new Uint8Array(new ArrayBuffer(1));
      // Hard check: a Pyodide release without setInterruptBuffer must fail loudly here.
      if (typeof pyodide.setInterruptBuffer !== "function") {
        throw new Error(
          `Pyodide ${PYODIDE_VERSION} has no setInterruptBuffer, so a running cell could not `
          + `be cancelled. Pin a release that provides it.`,
        );
      }
      pyodide.setInterruptBuffer(sentinel);

      runtimeHandle = { pyodide, interruptBuffer: sentinel };
      onStatus?.({ kind: "ready" });
      return runtimeHandle;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      onStatus?.({ kind: "error", message: msg });
      runtimeLoading = null; // allow retry
      throw e;
    }
  })();
  return runtimeLoading;
}

/**
 * Load a Pyodide package on demand (call after detecting ModuleNotFoundError).
 * Throws MissingPackageError when the package is not in the curated index.
 */
export async function loadPyodidePackage(
  importName: string,
  onStatus?: PyodideStatusListener,
): Promise<void> {
  if (!runtimeHandle) {
    throw new Error("Pyodide runtime not booted yet — run a cell first");
  }
  const pkg = packageNameForImport(importName);
  onStatus?.({ kind: "loading-package", name: pkg });
  try {
    await runtimeHandle.pyodide.loadPackage([pkg]);
    onStatus?.({ kind: "ready" });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    onStatus?.({ kind: "error", message: msg });
    throw new MissingPackageError(pkg, msg);
  }
}

/**
 * Thrown when the package isn't in Pyodide's curated index.
 * The UI catches this to offer the micropip fallback.
 */
export class MissingPackageError extends Error {
  // Explicit fields (not constructor params) so Node's strip-type pass doesn't reject them.
  readonly pkg: string;
  readonly underlying: string;

  constructor(pkg: string, underlying: string) {
    super(
      `Pyodide doesn't ship a '${pkg}' package. Try the micropip fallback for pure-Python wheels on PyPI.`,
    );
    this.name = "MissingPackageError";
    this.pkg = pkg;
    this.underlying = underlying;
  }
}

/**
 * Install via micropip — fallback for PyPI pure-Python wheels not in the curated index.
 * Throws when no pure-Python wheel exists on PyPI (C-extension wheels need the lockfile).
 */
export async function micropipInstall(
  importName: string,
  onStatus?: PyodideStatusListener,
): Promise<void> {
  if (!runtimeHandle) {
    throw new Error("Pyodide runtime not booted yet — run a cell first");
  }
  const pkg = packageNameForImport(importName);
  onStatus?.({ kind: "loading-package", name: "micropip" });
  try {
    await runtimeHandle.pyodide.loadPackage(["micropip"]);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    onStatus?.({ kind: "error", message: msg });
    throw new Error(`Failed to bootstrap micropip: ${msg}`);
  }
  onStatus?.({ kind: "loading-package", name: pkg });
  // Pass through globals.set to prevent injection into the Python string literal.
  runtimeHandle.pyodide.globals.set("__nimbus_pkg__", pkg);
  try {
    await runtimeHandle.pyodide.runPythonAsync(
      "import micropip\nawait micropip.install(__nimbus_pkg__)",
    );
    onStatus?.({ kind: "ready" });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    onStatus?.({ kind: "error", message: msg });
    throw new Error(
      `micropip couldn't install '${pkg}': ${msg}. The package may not have a pure-Python wheel on PyPI, or your network blocks files.pythonhosted.org.`,
    );
  } finally {
    // Clean up the temporary global from user namespace.
    try {
      runtimeHandle.pyodide.runPython("del __nimbus_pkg__");
    } catch {
      // ignore
    }
  }
}

/** Package names currently loaded. Empty before boot (no interpreter = nothing loaded). */
export function loadedPackageNames(): ReadonlySet<string> {
  if (!runtimeHandle) return new Set();
  const loaded = runtimeHandle.pyodide.loadedPackages;
  return new Set(loaded ? Object.keys(loaded) : []);
}

/**
 * Install by name via the appropriate route (curated index or micropip).
 * Boots the runtime if needed so the packages panel can install before any cell runs.
 */
export async function installPackage(
  importName: string,
  onStatus?: PyodideStatusListener,
): Promise<void> {
  await getPyodide(onStatus);
  if (isBundledPyodideImport(importName)) {
    await loadPyodidePackage(importName, onStatus);
    return;
  }
  // Not in the lockfile; micropip is the only remaining route.
  await micropipInstall(importName, onStatus);
}

/**
 * Fetch Pyodide's ESM loader. Dynamic import used because this runs inside a Worker
 * (no document). @vite-ignore: URL is CDN or vendored path resolved at runtime.
 */
async function loadPyodideModule(): Promise<(options: any) => Promise<any>> {
  const url = `${PYODIDE_BASE_URL}pyodide.mjs`;
  let loaded: any;
  try {
    loaded = await import(/* @vite-ignore */ url);
  } catch (e) {
    const detail = e instanceof Error ? e.message : String(e);
    throw new Error(
      `Pyodide loader failed to download (${url}): ${detail}. Check network connectivity.`,
    );
  }
  const loader = loaded?.loadPyodide ?? loaded?.default?.loadPyodide;
  if (typeof loader !== "function") {
    throw new Error(`Pyodide loader missing from ${url}`);
  }
  return loader;
}

/* ---------- Field injection ---------- */

/** Push field data into Python globals as numpy arrays before each cell execution. */
export async function bindContext(
  handle: PyodideHandle,
  ctx: InjectionContext,
): Promise<void> {
  const { pyodide } = handle;
  // Always import numpy so user code can rely on it being available.
  pyodide.runPython("import numpy as np");

  // Field globals belong to the frame binding, not to the persistent user
  // namespace. Remove every field Nimbus owned last time before admitting the
  // new set: otherwise a budget-skipped or no-longer-exported field silently
  // remains available with values from an older frame.
  pyodide.runPython(`
for _nimbus_old_field in list(globals().get("_bound_fields", [])):
    globals().pop(str(_nimbus_old_field), None)
globals().pop("_nimbus_old_field", None)
`);

  const boundShape = ctx.fields[0]?.field.shape;
  const shape: [number, number, number] = [ctx.grid?.nr ?? boundShape?.[0] ?? 0,
    ctx.grid?.nth ?? boundShape?.[1] ?? 0, ctx.grid?.nz ?? boundShape?.[2] ?? 0];
  const coordinates = pythonFieldCoordinates(ctx.grid, shape, shape, [0, 0, 0], [1, 1, 1]);
  if (coordinates[2]?.unit === "model_level" && ctx.modelLevelIndices && ctx.modelLevelIndices.length === shape[2]) {
    coordinates[2] = { ...coordinates[2], values: [...ctx.modelLevelIndices] };
  }
  pyodide.globals.set("nr", shape[0]);
  pyodide.globals.set("nth", shape[1]);
  pyodide.globals.set("nz", shape[2]);
  pyodide.globals.set("dr_m", coordinates[0]?.unit === "m" ? ctx.grid?.drM ?? Number.NaN : Number.NaN);
  pyodide.globals.set("dtheta_rad", coordinates[1]?.unit === "rad" ? ctx.grid?.dthetaRad ?? Number.NaN : 0);
  pyodide.globals.set("dy_m", coordinates[1]?.unit === "m" ? ctx.grid?.dyM ?? ctx.grid?.drM ?? Number.NaN : Number.NaN);
  pyodide.globals.set("coordinate_system", ctx.grid?.coordinateSystem ?? "unknown");
  pyodide.globals.set("terrain_following", ctx.grid?.terrainFollowing ?? false);
  pyodide.globals.set("dz_m", ctx.grid?.terrainFollowing ? 0
    : coordinates[2]?.unit === "m" ? ctx.grid?.dzM ?? Number.NaN : Number.NaN);
  pyodide.globals.set("simulation_time_s", ctx.simulationTimeS ?? Number.NaN);
  pyodide.globals.set("_scalar_origin_m", coordinates.map(axis => axis.unit === "m" ? axis.values[0] ?? Number.NaN : Number.NaN));
  pyodide.globals.set("_coordinate_units", coordinates.map(axis => axis.unit));
  for (const [index, name] of ["x", "y", "z"].entries()) {
    const axis = coordinates[index];
    pyodide.globals.set(`_${name}_coordinates`, axis?.values ?? []);
    pyodide.globals.set(`_${name}_m`, axis?.unit === "m" ? axis.values : []);
    pyodide.globals.set(`_${name}_index`, index === 2 && ctx.modelLevelIndices?.length === shape[2]
      ? ctx.modelLevelIndices : Array.from({ length: shape[index] ?? 0 }, (_, cell) => cell));
  }
  pyodide.globals.set("_theta_rad", coordinates[1]?.unit === "rad" ? coordinates[1].values : []);
  pyodide.globals.set("_valid_lower", ctx.grid?.validIndexRange?.lower ?? [0, 0, 0]);
  pyodide.globals.set("_valid_upper", ctx.grid?.validIndexRange?.upper ?? shape.map(size => Math.max(0, size - 1)));
  pyodide.runPython(`
scalar_origin_m = np.asarray(_scalar_origin_m, dtype=np.float64)
valid_lower = np.asarray(_valid_lower, dtype=np.int64)
valid_upper = np.asarray(_valid_upper, dtype=np.int64)
coordinate_units = tuple(_coordinate_units.to_py() if hasattr(_coordinate_units, "to_py") else _coordinate_units)
x_coordinates = np.asarray(_x_coordinates, dtype=np.float64)
y_coordinates = np.asarray(_y_coordinates, dtype=np.float64)
z_coordinates = np.asarray(_z_coordinates, dtype=np.float64)
x_m = np.asarray(_x_m, dtype=np.float64)
y_m = np.asarray(_y_m, dtype=np.float64)
z_m = np.asarray(_z_m, dtype=np.float64)
theta_rad = np.asarray(_theta_rad, dtype=np.float64)
x_index = np.asarray(_x_index, dtype=np.int64)
y_index = np.asarray(_y_index, dtype=np.int64)
z_index = np.asarray(_z_index, dtype=np.int64)
`);

  for (const { name, field } of ctx.fields) {
    pyodide.globals.set(`_${name}_buffer`, field.data);
    pyodide.globals.set(`_${name}_shape`, field.shape);
    // np.asarray copies from JS heap to wasm memory. Drop temporaries immediately
    // to avoid holding a second copy of the full field across the session.
    try {
      pyodide.runPython(
        `${name} = np.asarray(_${name}_buffer, dtype=np.float32)`
        + `.reshape(tuple(_${name}_shape.to_py()))\n`
        + `del _${name}_buffer, _${name}_shape`,
      );
    } catch (e) {
      // Surface a human-readable message; the raw MemoryError traceback names no field.
      const mib = Math.round(field.data.byteLength / (1024 * 1024));
      throw new Error(
        `The interpreter ran out of memory binding '${name}' (${mib} MiB). A field is `
        + `copied whole into a 32-bit heap, so a large domain does not fit several times `
        + `over. Reference fewer fields in your code, or load one at a time and reduce it `
        + `before the next: arr = await load_field("${name}", stride=(4, 4, 2)). `
        + `Underlying error: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  }

  pyodide.globals.set("frame_index", ctx.frameIndex === null ? null : ctx.frameIndex ?? 0);
  pyodide.globals.set("step_index", ctx.stepIndex ?? ctx.frameIndex ?? 0);
  pyodide.globals.set("frame_count", ctx.frameCount ?? 1);
  pyodide.globals.set("_frame_times_s", [...(ctx.frameTimesS ?? [])]);
  pyodide.runPython("frame_times_s = np.asarray(_frame_times_s, dtype=np.float64)");
  pyodide.globals.set(
    "_sources",
    (ctx.sources ?? [{ role: "primary", displayName: "primary", frameCount: ctx.frameCount ?? 1 }])
      .map((source) => ({ ...source })),
  );
  pyodide.runPython(`
datasets = {
    entry["role"]: {"name": entry["displayName"], "frames": entry["frameCount"]}
    for entry in (_sources.to_py() if hasattr(_sources, "to_py") else _sources)
}
`);
  bindFrameLoader(pyodide, ctx.loadFrameField);
  pyodide.globals.set("_bound_fields", ctx.fields.map(({ name }) => name));
  pyodide.globals.set("_field_units", ctx.fieldUnits ? { ...ctx.fieldUnits } : {});
  // JsProxy → native Python; normalise once so dataset() needn't call to_py everywhere.
  pyodide.runPython(`
_bound_fields = _bound_fields.to_py() if hasattr(_bound_fields, "to_py") else list(_bound_fields)
_field_units = _field_units.to_py() if hasattr(_field_units, "to_py") else dict(_field_units)
`);
  pyodide.runPython(DATASET_HELPER);
}

// display() protocol: appends to a per-run list; result/last-open-figure fallback appended at end.
// Arrays cross the FFI via globals keyed by sequence number (same route as field injection).
const DISPLAY_HELPER = `
import base64 as _b64, io as _io, json as _json, os as _os

_NIMBUS_MAX_ANIMATION_FRAMES = ${MAX_ANIMATION_FRAMES}
_NIMBUS_MAX_OUTPUTS = ${MAX_OUTPUTS_PER_RUN}
_NIMBUS_MAX_COMPLETIONS = ${MAX_COMPLETIONS}
_NIMBUS_MAX_OUTPUT_ITEM_BYTES = ${MAX_OUTPUT_ITEM_BYTES}
_NIMBUS_MAX_OUTPUT_TOTAL_BYTES = ${MAX_OUTPUT_TOTAL_BYTES}
_NIMBUS_MAX_HTML_OUTPUT_BYTES = ${MAX_HTML_OUTPUT_BYTES}
_NIMBUS_MAX_ANIMATION_OUTPUT_BYTES = ${MAX_ANIMATION_OUTPUT_BYTES}
# Distinct from None so that an attribute whose value IS None is reported as
# found rather than as a broken chain.
_NIMBUS_MISSING = object()
_NIMBUS_OUT_DIR = ${JSON.stringify(PYODIDE_OUTPUT_DIR)}


class NimbusAnimation:
    """A sequence of rendered frames, playable inline and exportable."""

    def __init__(self, frames, fps=10):
        self.frames = list(frames)
        self.fps = max(1, min(60, int(fps)))

    def __len__(self):
        return len(self.frames)

    def __repr__(self):
        return f"NimbusAnimation({len(self.frames)} frames @ {self.fps} fps)"


def _nimbus_figure_bounds(figure):
    dpi = float(figure.dpi)
    width, height = figure.get_size_inches()
    def check(w, h):
        if not np.isfinite([w, h, dpi]).all() or min(w, h, dpi) <= 0 or max(w * dpi, h * dpi) > 8192 or w * h * dpi * dpi > 16_000_000:
            raise ValueError("Figure exceeds the 16 megapixel / 8192 pixel-side capture limit; reduce figsize or dpi")
    check(width, height)
    figure.canvas.draw()
    bounds = figure.get_tightbbox(figure.canvas.get_renderer()).padded(0.1)
    check(bounds.width, bounds.height)
    return bounds


def _nimbus_figure_png(figure):
    bounds = _nimbus_figure_bounds(figure)
    buffer = _io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches=bounds, dpi=figure.dpi)
    return _b64.b64encode(buffer.getvalue()).decode("ascii")


def _nimbus_figure_svg(figure):
    buffer = _io.StringIO()
    figure.savefig(buffer, format="svg", bbox_inches="tight")
    return buffer.getvalue()


def animate(frames, fps=10):
    """Build a playable animation from matplotlib figures or 2-D arrays.

    Each frame is rendered immediately, so the result is self-contained and can
    be exported to GIF or MP4 without re-running your code.
    """
    import matplotlib.pyplot as _plt

    rendered = []
    rendered_bytes = 0
    for item in frames:
        if len(rendered) >= _NIMBUS_MAX_ANIMATION_FRAMES:
            raise ValueError(
                f"animate() is limited to {_NIMBUS_MAX_ANIMATION_FRAMES} frames; "
                "subsample the sequence"
            )
        if hasattr(item, "savefig"):
            encoded = _nimbus_figure_png(item)
            _plt.close(item)
        else:
            array = np.asarray(item)
            if array.ndim != 2:
                raise ValueError(
                    "animate() takes matplotlib figures or 2-D arrays; "
                    f"got an array with {array.ndim} dimensions"
                )
            figure, axes = _plt.subplots()
            axes.imshow(array.T, origin="lower")
            axes.set_axis_off()
            encoded = _nimbus_figure_png(figure)
            _plt.close(figure)
        rendered_bytes += len(encoded)
        if rendered_bytes > _NIMBUS_MAX_ANIMATION_OUTPUT_BYTES:
            raise ValueError(
                "animate() encoded output exceeds the 128 MiB transfer limit; "
                "use fewer frames, a smaller figure, or lower DPI"
            )
        rendered.append(encoded)
    if not rendered:
        raise ValueError("animate() needs at least one frame")
    return NimbusAnimation(rendered, fps=fps)


def save_path(name):
    """A path under the export directory. Write here to offer a file for saving.

    e.g. np.save(save_path("w.npy"), w) then use Save in the Files panel.
    """
    _os.makedirs(_NIMBUS_OUT_DIR, exist_ok=True)
    name = str(name)
    if not name or name in (".", "..") or "/" in name or "\\\\" in name or "\\x00" in name:
        raise ValueError("save_path() takes a bare file name, not a path")
    return f"{_NIMBUS_OUT_DIR}/{name}"


def _nimbus_capture(value, label=None):
    entry = {"label": label}
    # An animation is checked before anything else: matplotlib's own
    # FuncAnimation also satisfies several of the tests below.
    if isinstance(value, NimbusAnimation):
        entry["kind"] = "animation"
        entry["frames"] = value.frames
        entry["fps"] = value.fps
        entry["bytes"] = sum(len(frame) for frame in value.frames)
    elif hasattr(value, "to_jshtml"):
        # matplotlib FuncAnimation: its own player is self-contained HTML.
        entry["kind"] = "html"
        entry["html"] = value.to_jshtml()
        entry["scripts"] = True
        entry["bytes"] = len(entry["html"].encode("utf-8"))
    elif hasattr(value, "savefig"):
        entry["kind"] = "image"
        entry["png"] = _nimbus_figure_png(value)
        entry["svg"] = _nimbus_figure_svg(value)
        entry["bytes"] = len(entry["png"]) + len(entry["svg"].encode("utf-8"))
    elif isinstance(value, np.ndarray):
        global _nimbus_output_sequence
        _nimbus_output_sequence += 1
        key = f"_nimbus_out_{_nimbus_output_sequence}"
        if value.dtype.kind not in "biuf" or value.dtype.itemsize not in (1, 2, 4, 8) or (value.dtype.kind == "f" and value.dtype.itemsize not in (4, 8)):
            raise ValueError("display() cannot plot this dtype without conversion; choose a real numeric view explicitly, or use np.save('/nimbus/out/array.npy', array) to preserve the original array")
        entry["bytes"] = int(value.nbytes)
        if entry["bytes"] > _NIMBUS_MAX_OUTPUT_ITEM_BYTES:
            raise ValueError(
                f"displayed array is {entry['bytes'] / (1024 * 1024):.1f} MiB; "
                "slice or reduce it below the 64 MiB per-output limit"
            )
        converted = np.array(value, dtype=value.dtype.newbyteorder("<"), order="C", copy=True)
        entry["dtype"] = converted.dtype.str
        globals()[key] = converted
        entry["kind"] = "ndarray"
        entry["key"] = key
        entry["shape"] = list(value.shape)
    elif hasattr(value, "_repr_html_"):
        # xarray and pandas both render themselves far better than repr() does.
        entry["kind"] = "html"
        entry["html"] = value._repr_html_()
        entry["scripts"] = False
        entry["bytes"] = len(entry["html"].encode("utf-8"))
    elif isinstance(value, (int, float, bool, str)):
        entry["kind"] = "text"
        entry["text"] = str(value)
        entry["bytes"] = len(entry["text"].encode("utf-8"))
    else:
        entry["kind"] = "text"
        entry["text"] = repr(value)
        entry["bytes"] = len(entry["text"].encode("utf-8"))
    limit = (_NIMBUS_MAX_ANIMATION_OUTPUT_BYTES if entry["kind"] == "animation"
             else _NIMBUS_MAX_HTML_OUTPUT_BYTES if entry["kind"] == "html"
             else _NIMBUS_MAX_OUTPUT_ITEM_BYTES)
    if entry.get("bytes", 0) > limit:
        raise ValueError(
            f"display output is {entry['bytes'] / (1024 * 1024):.1f} MiB, above "
            f"the {limit / (1024 * 1024):.0f} MiB limit for {entry['kind']} output"
        )
    return entry


def display(*values, label=None):
    """Show one or more values. Call as often as you like within a cell."""
    global _nimbus_output_bytes
    for value in values:
        if len(_nimbus_outputs) >= _NIMBUS_MAX_OUTPUTS:
            raise ValueError(f"a cell may display at most {_NIMBUS_MAX_OUTPUTS} outputs")
        captured = _nimbus_capture(value, label)
        next_total = _nimbus_output_bytes + int(captured.get("bytes", 0))
        if next_total > _NIMBUS_MAX_OUTPUT_TOTAL_BYTES:
            key = captured.get("key")
            if key is not None:
                globals().pop(key, None)
            raise ValueError(
                "cell display output exceeds the 256 MiB total transfer limit; "
                "show fewer values or reduce them first"
            )
        _nimbus_output_bytes = next_total
        _nimbus_outputs.append(captured)


def _nimbus_complete(path, partial):
    """Candidate names for the editor's completion popup.

    Resolution is static: requesting completion must not invoke a user's
    __dir__, __getattr__, property, or descriptor merely because they typed a dot.

    Kinds and signatures are computed here rather than in the UI because this
    is the only side that can see the objects.
    """
    import builtins as _builtins
    import inspect as _inspect
    import keyword as _keyword

    def static_members(target):
        members = {}
        try:
            namespace = _inspect.getattr_static(target, "__dict__", {})
            if hasattr(namespace, "items"):
                members.update(namespace)
        except Exception:
            pass
        target_type = target if _inspect.isclass(target) else type(target)
        for cls in getattr(target_type, "__mro__", ()):
            try:
                members.update(type.__getattribute__(cls, "__dict__"))
            except Exception:
                pass
        return members

    if path:
        target = globals().get(path[0], vars(_builtins).get(path[0], _NIMBUS_MISSING))
        for step in path[1:]:
            if target is _NIMBUS_MISSING:
                break
            target = static_members(target).get(step, _NIMBUS_MISSING)
        if target is _NIMBUS_MISSING:
            return _json.dumps([])
        members = static_members(target)
        names = members
    else:
        names = list(globals()) + dir(_builtins) + _keyword.kwlist
        members = None

    seen = set()
    out = []
    for name in names:
        if name in seen or not name.startswith(partial):
            continue
        seen.add(name)
        if members is None and name in _keyword.kwlist:
            out.append({"label": name, "kind": "keyword", "detail": None})
            continue
        value = (members.get(name, _NIMBUS_MISSING) if members is not None
                 else globals().get(name, vars(_builtins).get(name, _NIMBUS_MISSING)))
        if value is _NIMBUS_MISSING:
            continue
        out.append({"label": name, "kind": _nimbus_kind(value), "detail": _nimbus_detail(value)})
        if len(out) >= _NIMBUS_MAX_COMPLETIONS:
            break
    return _json.dumps(out)


# Names this module injects, hidden from the namespace panel. The panel answers
# "what have I got", and a list where the user's three variables sit among two
# dozen helpers does not answer it; the helpers are documented in the starter
# script instead. "result" is deliberately NOT here -- the user assigns it, so
# hiding it would conceal their own binding. (No backticks in this string: it
# is a JS template literal, and one would end it early.)
_NIMBUS_OWN_NAMES = frozenset({
    "np", "display", "animate", "save_path", "tick", "dataset",
    "load_field", "field_metadata", "load_series", "load_dataset", "files", "datasets",
    "NimbusAnimation",
})


def _nimbus_namespace():
    """The user's own bindings, as JSON."""
    out = []
    for name, value in sorted(globals().items()):
        # Underscore-prefixed names are this module's plumbing and Python's own
        # (__builtins__, __name__); neither is the user's.
        if name.startswith("_") or name in _NIMBUS_OWN_NAMES:
            continue
        try:
            kind = _nimbus_kind(value)
            detail = _nimbus_detail(value)
        except Exception:
            kind, detail = "variable", None
        out.append({"name": name, "kind": kind, "detail": detail})
    return _json.dumps(out)


def _nimbus_kind(value):
    import inspect as _inspect

    if _inspect.ismodule(value):
        return "module"
    if _inspect.isclass(value):
        return "class"
    if callable(value):
        return "function"
    return "variable"


def _nimbus_detail(value):
    import inspect as _inspect

    # inspect.signature consults arbitrary __signature__ attributes on general
    # callables. Restrict it to concrete Python functions and builtins.
    if _inspect.isfunction(value) or _inspect.isbuiltin(value):
        try:
            return f"{getattr(value, '__name__', '')}{_inspect.signature(value)}"
        except (TypeError, ValueError):
            # Builtins and C extensions frequently expose no signature. The
            # type is still more informative than nothing.
            return type(value).__name__
    if isinstance(value, np.ndarray):
        shape = value.shape
        return f"{type(value).__name__}{tuple(shape)}"
    return type(value).__name__


async def tick(seconds=0):
    """Yield, so printed output can appear and Stop can land.

    The interpreter has its own thread, so the window stays responsive whatever
    a loop does — but a loop that never awaits also never reaches the point
    where a KeyboardInterrupt can be raised, which is what makes Stop fall back
    to replacing the interpreter and losing the namespace. Awaiting this inside
    a long loop keeps the console current and keeps the cell interruptible:

        for i in range(200):
            heavy_step(i)
            print("step", i)
            await tick()
    """
    import asyncio as _asyncio

    await _asyncio.sleep(seconds)
`;

/**
 * Completion candidates for `path` + `partial`. Returns empty if no interpreter
 * or cell is in flight (must never boot Pyodide or interleave with a running cell).
 */
export function completeNames(
  path: readonly string[],
  partial: string,
): readonly CompletionCandidate[] {
  if (!runtimeHandle || cellInFlight) return [];
  const { pyodide } = runtimeHandle;
  try {
    // Arguments via globals, not string interpolation — editor input could otherwise inject Python.
    pyodide.globals.set("__nimbus_complete_path__", pyodide.toPy([...path]));
    pyodide.globals.set("__nimbus_complete_partial__", partial);
    const json: string = pyodide.runPython(
      "_nimbus_complete(list(__nimbus_complete_path__), __nimbus_complete_partial__)",
    );
    const parsed = JSON.parse(json) as CompletionCandidate[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return []; // completion is an assist; failures must not surface
  } finally {
    try {
      pyodide.runPython(
        "del __nimbus_complete_path__\ndel __nimbus_complete_partial__",
      );
    } catch {
      // ignore
    }
  }
}

/**
 * Whether the interpreter can still execute. Clears the interrupt sentinel first
 * (a stopped cell leaves it set, which would falsely report the interpreter dead).
 * Probes with "1" because a wasm abort raises unrelated errors — message text is unreliable.
 */
export function interpreterIsResponsive(): boolean {
  if (!runtimeHandle) return false;
  try {
    runtimeHandle.interruptBuffer[0] = 0;
    runtimeHandle.pyodide.runPython("1");
    return true;
  } catch {
    return false;
  }
}

/**
 * Drop the dead interpreter. Does NOT boot a replacement — recovery means
 * replacing the whole worker (held wasm memory cannot be freed from this thread).
 */
export function discardInterpreter(): void {
  runtimeHandle = null;
  runtimeLoading = null;
  cellInFlight = false;
}

/**
 * Reserved wasm linear memory (high-water mark — never returned to host), or null before boot.
 * Matters because wasm32 allocation failures abort the interpreter rather than raising MemoryError.
 */
export function interpreterHeapBytes(): number | null {
  if (!runtimeHandle) return null;
  const heap = runtimeHandle.pyodide._module?.HEAPU8;
  return heap ? heap.length : null;
}

/** The user's current namespace (shared across all cells). */
export function inspectNamespace(): readonly NamespaceEntry[] {
  if (!runtimeHandle || cellInFlight) return [];
  try {
    const json: string = runtimeHandle.pyodide.runPython("_nimbus_namespace()");
    const parsed = JSON.parse(json) as NamespaceEntry[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

/** Counter for the temporary globals the frame loader hands across the FFI. */
let frameLoadSequence = 0;

// load_field bridge: only the current frame's fields are globals; this lets user
// code pull any frame on demand. Array crosses FFI via JS globals + np.asarray.
function bindFrameLoader(
  pyodide: any,
  loadFrameField: InjectionContext["loadFrameField"],
): void {
  if (!loadFrameField) {
    // Defining a helper that always raises would be worse than not defining it:
    // `load_field` would autocomplete and read as supported.
    pyodide.runPython("globals().pop('load_field', None)");
    return;
  }
  pyodide.globals.set("_load_frame_field", async (
    fieldName: string,
    frameIndex: number,
    source: string,
    stride: unknown,
    box: unknown,
    validOnly: boolean,
    terrainRemap: boolean,
  ) => {
    // JsProxy of a Python tuple is not a JS array; normalise to plain arrays here.
    const toPlain = (value: unknown): unknown => {
      if (value === null || value === undefined) return null;
      const proxied = value as { toJs?: () => unknown };
      return typeof proxied.toJs === "function" ? proxied.toJs() : value;
    };
    const stridePlain = toPlain(stride);
    const strides = Array.isArray(stridePlain) ? stridePlain.map(Number) : null;
    const boxPlain = toPlain(box);
    const bounds = Array.isArray(boxPlain)
      ? boxPlain.map((pair) => {
        const edge = toPlain(pair);
        const numbers = Array.isArray(edge) ? edge.map(Number) : [0, 0];
        return [numbers[0], numbers[1]] as [number, number];
      })
      : null;
    const field = await loadFrameField(fieldName, frameIndex, source, {
      stride: strides ?? [],
      box: bounds,
      validOnly,
      terrainRemap,
    });
    frameLoadSequence += 1;
    const key = `_frame_field_${frameLoadSequence}`;
    const bridgeKeys = [`${key}_buffer`, `${key}_shape`, `${key}_metadata`, key];
    try {
      pyodide.globals.set(`${key}_buffer`, field.data);
      pyodide.globals.set(`${key}_shape`, field.shape);
      pyodide.globals.set(`${key}_metadata`, field.pythonMetadata ?? null);
      pyodide.runPython(
        `${key} = np.asarray(${key}_buffer, dtype=np.float32).reshape(tuple(${key}_shape.to_py()))`,
      );
      return key;
    } catch (error) {
      for (const bridgeKey of bridgeKeys) {
        try {
          pyodide.globals.delete(bridgeKey);
        } catch {
          // Preserve the conversion error; cleanup is best-effort.
        }
      }
      throw error;
    }
  });
  pyodide.runPython(FRAME_LOADER_HELPER);
}

const FRAME_LOADER_HELPER = `
_NIMBUS_MAX_SERIES_BYTES = ${MAX_SERIES_BYTES}
_NIMBUS_MAX_DATASET_BYTES = ${MAX_DATASET_BYTES}


def _nimbus_mib(value):
    """Byte count as MiB, for a refusal that has to state two sizes."""
    return round(value / (1024 * 1024), 1)


_loaded_field_metadata = {}
_NIMBUS_MAX_LOADED_FIELD_METADATA = 256


async def load_field(name, frame=None, source="primary", stride=1, box=None,
                     valid_only=True, terrain_remap=False):
    """Load one field from any frame of any open dataset.

    Await it: \`w5 = await load_field("w", frame=5)\`. Pass source="comparison"
    to read the second dataset Studio has open — see \`datasets\`. Reads go
    through the same budgeted native path the plots use, so a large field takes
    real time.

    \`stride\` and \`box\` reduce the field BEFORE it reaches this interpreter,
    which is what makes a domain of any size workable. The interpreter is
    32-bit and holds a field whole, so a 512x512x160 run is 160 MiB a field and
    a few of them is all it can take — but the reduction happens on the reader
    side, out of that memory entirely, so what arrives is only what you asked
    for:

        coarse = await load_field("w", stride=4)          # valid cells, every 4th cell
        fine   = await load_field("w", box=((200, 312), (200, 312), (0, 40)))

    \`stride\` is one number for every axis or one per axis. \`box\` is a
    half-open (start, stop) per axis, applied before the stride, and is clamped
    to the field rather than refused. By default the request is also intersected
    with the simulator's valid (halo-free) cell range; pass \`valid_only=False\`
    only when you intentionally need storage/ghost cells. Neither changes what the data means — the
    spacing between the cells you get back is the file's spacing times the
    stride, which is what to hand np.gradient. On a terrain-following source,
    terrain_remap=True changes stride to physical-volume block reduction and
    conservatively remaps those blocks onto separable physical z coordinates.
    Use field_metadata(name) to retrieve the resulting coordinate axes.
    """
    if source not in datasets:
        available = ", ".join(sorted(datasets)) or "none"
        raise KeyError(f"unknown source '{source}'; open datasets: {available}")
    limit = datasets[source]["frames"]
    # Frame indices are per-source: the comparison dataset is frequently a
    # different length, so validating against the primary's count would accept
    # a frame that does not exist over there.
    target = frame_index if frame is None else int(frame)
    if not (0 <= target < limit):
        raise IndexError(f"frame {target} is outside 0..{limit - 1} for source '{source}'")
    # One number means the same on every axis. Sent as a plain list either way,
    # because a tuple crosses as a proxy the other side has to unwrap.
    steps = [int(stride)] * 3 if isinstance(stride, int) else [int(s) for s in stride]
    bounds = None if box is None else [[int(edge[0]), int(edge[1])] for edge in box]
    key = await _load_frame_field(
        name, target, source, steps, bounds, bool(valid_only), bool(terrain_remap)
    )
    # The bridge parks the array in a uniquely named global; move it into a
    # local and drop the global so a long session does not retain every frame
    # it has ever touched.
    array = globals().pop(key)
    globals().pop(key + "_buffer", None)
    globals().pop(key + "_shape", None)
    metadata = globals().pop(key + "_metadata", None)
    if hasattr(metadata, "to_py"):
        metadata = metadata.to_py()
    metadata_key = (source, target, str(name))
    if metadata_key not in _loaded_field_metadata and len(_loaded_field_metadata) >= _NIMBUS_MAX_LOADED_FIELD_METADATA:
        # dicts preserve insertion order. Keep recent coordinate/provenance
        # records without retaining one copy for every frame ever inspected.
        _loaded_field_metadata.pop(next(iter(_loaded_field_metadata)))
    _loaded_field_metadata[metadata_key] = metadata
    return array


def field_metadata(name, frame=None, source="primary"):
    """Coordinate, unit, and remap metadata for the most recent matching load."""
    target = frame_index if frame is None else int(frame)
    metadata = _loaded_field_metadata.get((source, target, str(name)))
    if metadata is None:
        raise KeyError(f"no metadata for '{name}'; await load_field() first")
    return metadata


async def load_series(name, frames=None, source="primary", stride=1, box=None,
                      valid_only=True, terrain_remap=False):
    """Stack one field across frames: shape (len(frames), *field.shape).

    Loads frames one at a time and holds them all, so the result is the field
    size times the frame count. Refused above a budget rather than attempted:
    overrunning this interpreter's memory aborts it outright instead of raising
    something you could catch, so the whole run of a large field is a request
    that ends the session. Subset the frames, or slice a level per frame.
    """
    if source not in datasets:
        available = ", ".join(sorted(datasets)) or "none"
        raise KeyError(f"unknown source '{source}'; open datasets: {available}")
    span = range(datasets[source]["frames"])
    targets = list(span) if frames is None else [int(f) for f in frames]
    if not targets:
        raise ValueError("load_series needs at least one frame")
    # One frame first, to learn the size from the data rather than from a
    # manifest that may not describe this field. It is a frame the caller asked
    # for either way, so nothing is read that the request did not already need.
    first = await load_field(
        name, frame=targets[0], source=source, stride=stride, box=box,
        valid_only=valid_only, terrain_remap=terrain_remap,
    )
    total = int(first.nbytes) * len(targets)
    if total > _NIMBUS_MAX_SERIES_BYTES:
        affordable = max(1, _NIMBUS_MAX_SERIES_BYTES // int(first.nbytes))
        raise MemoryError(
            f"load_series('{name}') over {len(targets)} frames would build "
            f"{_nimbus_mib(total)} MiB, above the {_nimbus_mib(_NIMBUS_MAX_SERIES_BYTES)} MiB "
            f"limit; about {affordable} frames of this field fit. Subset the frames, or "
            f"reduce each one as you go with a loop over load_field()."
        )
    shape = first.shape
    series = np.empty((len(targets), *shape), dtype=first.dtype)
    series[0] = first
    del first
    for index, target in enumerate(targets[1:], start=1):
        current = await load_field(
            name, frame=target, source=source, stride=stride, box=box,
            valid_only=valid_only, terrain_remap=terrain_remap,
        )
        if current.shape != shape:
            raise ValueError("load_series fields must have the same shape in every requested frame")
        series[index] = current
        del current
    return series


async def load_dataset(
    fields, frame=None, source="primary", stride=1, box=None, valid_only=True,
    terrain_remap=False
):
    """An xarray.Dataset of named fields from any frame of any open dataset.

    dataset() is the zero-cost view of the frame already on screen; this one
    reads, so it is async and takes an explicit field list — there is no
    manifest here to enumerate "everything" cheaply.
    """
    import xarray as xr

    names = list(fields)
    if not names:
        raise ValueError("load_dataset needs at least one field name")
    if len(set(names)) != len(names):
        raise ValueError("load_dataset field names must be unique")
    target = frame_index if frame is None else int(frame)
    arrays = {}
    retained_bytes = 0
    for name in names:
        array = await load_field(
            name, frame=target, source=source, stride=stride, box=box,
            valid_only=valid_only, terrain_remap=terrain_remap,
        )
        next_bytes = retained_bytes + int(array.nbytes)
        if next_bytes > _NIMBUS_MAX_DATASET_BYTES:
            raise MemoryError(
                f"load_dataset would retain {_nimbus_mib(next_bytes)} MiB after '{name}', "
                f"above the {_nimbus_mib(_NIMBUS_MAX_DATASET_BYTES)} MiB limit; "
                "request fewer fields, a tighter box, or a larger stride."
            )
        arrays[name] = array
        retained_bytes = next_bytes
    variables = {}
    shared_coords = {}
    for name, array in arrays.items():
        metadata = _loaded_field_metadata.get((source, target, str(name))) or {}
        coords_meta = metadata.get("coordinates", [])
        dims = tuple(entry["name"] for entry in coords_meta)
        if len(dims) != array.ndim:
            dims = ("x", "y", "z")[:array.ndim]
            coords = None
        else:
            coords = {
                entry["name"]: (
                    entry["name"], np.asarray(entry["values"], dtype=np.float64),
                    {"units": entry["unit"]},
                ) for entry in coords_meta
            }
        attrs = {"units": metadata["unit"]} if metadata.get("unit") else {}
        # Keep fidelity on each variable: terrain datasets can legitimately mix
        # linear-reflectivity and ordinary physical-volume means.
        attrs["nimbus_dtype"] = metadata.get("dtype", str(array.dtype))
        attrs["nimbus_sampling"] = metadata.get("sampling", "unknown")
        attrs["nimbus_aggregation"] = metadata.get("aggregation")
        attrs["nimbus_weighting"] = metadata.get("weighting")
        attrs["nimbus_source_format"] = metadata.get("sourceFormat")
        attrs["nimbus_compression"] = metadata.get("compression")
        attrs["nimbus_source_shape"] = metadata.get("sourceShape")
        attrs["nimbus_output_shape"] = metadata.get("outputShape", list(array.shape))
        attrs["nimbus_stride"] = metadata.get("stride")
        attrs["nimbus_box"] = metadata.get("box")
        variables[name] = (dims, array, attrs)
        if coords is not None:
            for dim, coordinate in coords.items():
                if dim in shared_coords and not np.array_equal(
                    coordinate[1], shared_coords[dim][1]
                ):
                    raise ValueError(
                        "requested fields do not share the same spatial coordinates"
                    )
                shared_coords[dim] = coordinate
    built = xr.Dataset(variables, coords=shared_coords)
    built.attrs["source"] = source
    built.attrs["source_name"] = datasets[source]["name"]
    built.attrs["frame_index"] = target
    metadata = _loaded_field_metadata.get((source, target, str(names[0]))) or {}
    built.attrs["step_index"] = metadata.get("stepIndex")
    built.attrs["simulation_time_s"] = metadata.get("simulationTimeS")
    built.attrs["coordinate_system"] = metadata.get("coordinateSystem", "unknown")
    built.attrs["terrain_following"] = bool(metadata.get("sourceTerrainFollowing", False))
    built.attrs["coordinate_resampling"] = metadata.get("coordinateResampling", "none")
    built.attrs["valid_only"] = bool(metadata.get("validOnly", valid_only))
    return built
`;

// dataset() — bound frame as an xarray.Dataset with named dims and physical coords.
// Bare `import xarray` is intentional: ModuleNotFoundError triggers the Load banner.
const DATASET_HELPER = `
def dataset(fields=None, valid_only=True):
    """Bound fields as an xarray.Dataset; omit storage halos by default."""
    import xarray as xr

    names = list(_bound_fields) if fields is None else list(fields)
    units = _field_units

    # A terrain-following frame has no separable physical height, so z_m is
    # empty by construction; fall back to model-level index rather than
    # inventing a vertical coordinate that does not exist.
    height_is_physical = z_m.size == nz and coordinate_units[2] == "m"
    lower = valid_lower if valid_only else np.zeros(3, dtype=np.int64)
    upper = valid_upper if valid_only else np.asarray([nr - 1, nth - 1, nz - 1])
    slices = tuple(slice(int(lower[axis]), int(upper[axis]) + 1) for axis in range(3))
    cylindrical = str(coordinate_system).lower().startswith("cyl")
    dims = ("r", "theta", "z") if cylindrical else ("x", "y", "z")
    coords = {
        dims[0]: (dims[0], x_coordinates[slices[0]], {"units": coordinate_units[0]}),
        dims[1]: (dims[1], y_coordinates[slices[1]], {"units": coordinate_units[1]}),
        dims[2]: (dims[2], (z_m if height_is_physical else z_coordinates)[slices[2]],
              {"units": coordinate_units[2]}),
    }

    variables = {}
    for name in names:
        array = globals().get(name)
        if array is None:
            continue
        attrs = {"units": units[name]} if name in units else {}
        if array.ndim == 3:
            variables[name] = (dims, array[slices], attrs)
        elif array.ndim == 2:
            variables[name] = (dims[:2], array[slices[:2]], attrs)
        # Anything else (scalars, rank-1 profiles) has no agreed dim mapping;
        # leaving it out beats guessing one.

    built = xr.Dataset(variables, coords=coords)
    built.attrs["simulation_time_s"] = simulation_time_s
    built.attrs["frame_index"] = frame_index
    built.attrs["step_index"] = step_index
    built.attrs["source"] = "primary"
    built.attrs["source_name"] = datasets.get("primary", {}).get("name", "primary")
    built.attrs["coordinate_system"] = coordinate_system
    built.attrs["terrain_following"] = bool(terrain_following)
    built.attrs["valid_only"] = bool(valid_only)
    return built
`;

/* ---------- Execution + result extraction ---------- */

/**
 * Fetch all packages this cell imports before it runs (batch vs one-per-failure).
 * Swallows errors: preload failure must not abort a cell that needed nothing,
 * and missing packages still surface via the cell's own ModuleNotFoundError.
 */
async function preloadImports(
  pyodide: any,
  source: string,
  onStatus?: PyodideStatusListener,
): Promise<void> {
  try {
    // dataset()/load_dataset() import xarray inside their helper bodies, which
    // Pyodide cannot discover by scanning the user's cell. Add the dependency
    // only when executable user code calls one of those helpers; comments and
    // strings must not make the heavy xarray stack a startup dependency.
    const usesDatasetHelper = /\b(?:load_dataset|dataset)\s*\(/.test(
      stripPythonNonCode(source),
    );
    const importSource = usesDatasetHelper ? `${source}\nimport xarray` : source;
    await pyodide.loadPackagesFromImports(importSource, {
      messageCallback: (message: string) => {
        onStatus?.({ kind: "loading-package", name: message });
      },
      // Keep Pyodide's lock-file SHA-256 verification enabled. Production loads
      // wheels from the CDN, so TLS alone is not a substitute for the pinned digest.
      checkIntegrity: true,
    });
    onStatus?.({ kind: "ready" });
  } catch {
    // See above: the import itself is the honest place for this to surface.
  }
}

/** Run a user cell. Captures stdout/stderr via StringIO redirect. */
export async function runPythonCell(
  handle: PyodideHandle,
  source: string,
  options: RunOptions = {},
  /* Worker-side only, so it is a parameter rather than part of RunOptions: a
   * callback cannot cross the postMessage boundary that RunOptions does. */
  onStatus?: PyodideStatusListener,
): Promise<RunResult> {
  const { pyodide } = handle;
  // Reset the interrupt buffer in case a previous cell tripped it.
  handle.interruptBuffer[0] = 0;

  await preloadImports(pyodide, source, onStatus);

  // Install sink before the redirect so the tee can reference it. None disables it.
  const sink = options.onStream;
  pyodide.globals.set(
    "_nimbus_stream_sink",
    sink
      ? (stream: string, text: string) => {
          if (stream === "stdout" || stream === "stderr") sink(stream, text);
        }
      : null,
  );

  // Source crosses through a global, never string interpolation. The setup uses
  // its AST only to decide whether this cell owns the conventional result output.
  pyodide.globals.set("__nimbus_cell_source__", source);

  // Redirect to StringIO; user source runs verbatim (not re-indented — that corrupts
  // triple-quoted strings and raises TabError on tab-indented code).
  pyodide.runPython(`
import io, sys
_NIMBUS_MAX_CAPTURED_STREAM_CHARS = ${MAX_CAPTURED_STREAM_CHARS}


class _NimbusCappedText:
    """StringIO-compatible final capture that cannot grow without bound."""

    def __init__(self):
        self._buffer = io.StringIO()
        self._length = 0
        self._truncated = False

    def write(self, text):
        text = str(text)
        remaining = _NIMBUS_MAX_CAPTURED_STREAM_CHARS - self._length
        if remaining > 0:
            kept = text[:remaining]
            self._buffer.write(kept)
            self._length += len(kept)
        if len(text) > remaining:
            self._truncated = True
        return len(text)

    def getvalue(self):
        value = self._buffer.getvalue()
        if self._truncated:
            value += "\\n[Nimbus: final stream capture truncated at 2 MiB]\\n"
        return value


_stdout = _NimbusCappedText()
_stderr = _NimbusCappedText()
__nimbus_user_result__ = None
_nimbus_outputs = []
_nimbus_output_bytes = 0
import ast as _nimbus_ast
try:
    _nimbus_result_assigned = any(
        isinstance(node, ( _nimbus_ast.Name, _nimbus_ast.arg))
        and getattr(node, "id", getattr(node, "arg", None)) == "result"
        and (not isinstance(node, _nimbus_ast.Name) or isinstance(node.ctx, _nimbus_ast.Store))
        for node in _nimbus_ast.walk(compile(
            __nimbus_cell_source__, "<exec>", "exec",
            flags=_nimbus_ast.PyCF_ONLY_AST | _nimbus_ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        ))
    )
except Exception:
    _nimbus_result_assigned = False


class _NimbusTee:
    """Writes to the captured buffer and mirrors the chunk to the UI.

    The buffer stays the source of truth for the run's final text; the mirror
    is best-effort and a failure in it must never surface as an error inside
    the user's cell, which is why the callback is guarded.
    """

    def __init__(self, buffer, sink, name):
        self._buffer = buffer
        self._sink = sink
        self._name = name

    def write(self, text):
        written = self._buffer.write(text)
        if self._sink is not None:
            try:
                self._sink(self._name, text)
            except Exception:
                pass
        return written

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        pass

    # Libraries branch on this to decide whether to emit progress bars and
    # colour codes. Reporting False keeps tqdm and friends on their plain
    # output, which is what a text pane can render.
    def isatty(self):
        return False


_orig_stdout, _orig_stderr = sys.stdout, sys.stderr
sys.stdout = _NimbusTee(_stdout, _nimbus_stream_sink, "stdout")
sys.stderr = _NimbusTee(_stderr, _nimbus_stream_sink, "stderr")
# Clear the previous run's exports so the Files panel shows what THIS run
# produced rather than an accumulating pile from the whole session. Skipped for
# a fragment that starts mid-document, where the earlier products are the ones
# the cells above just made.
import os as _os, shutil as _shutil
${options.resetProducedFiles === false
  ? ""
  : `_shutil.rmtree(${JSON.stringify(PYODIDE_OUTPUT_DIR)}, ignore_errors=True)\n`}_os.makedirs(${JSON.stringify(PYODIDE_OUTPUT_DIR)}, exist_ok=True)
# Discard figures left by an earlier cell. pyplot's registry is global, so
# without this a cell that draws nothing would be credited with the previous
# cell's figure and render a stale plot as if it were this run's output.
if "matplotlib.pyplot" in sys.modules:
    sys.modules["matplotlib.pyplot"].close("all")
`);

  // runPythonAsync supports top-level await; KeyboardInterrupt and exceptions go to catch.
  let succeeded = true;
  let exception: Error | null = null;
  cellInFlight = true;
  try {
    await pyodide.runPythonAsync(source);
  } catch (e) {
    succeeded = false;
    exception = e instanceof Error ? e : new Error(String(e));
  } finally {
    cellInFlight = false;
    // Always restore the real streams, even if the cell raised.
    try {
      pyodide.runPython("sys.stdout, sys.stderr = _orig_stdout, _orig_stderr");
    } catch {
      // ignore restore failures
    }
    // Sink closes over React state; must be cleared or it keeps setState alive in the interpreter.
    try {
      pyodide.globals.set("_nimbus_stream_sink", null);
    } catch {
      // ignore
    }
    try {
      pyodide.globals.delete("__nimbus_cell_source__");
    } catch {
      // ignore
    }
  }

  const stdout: string = pyodide.globals.get("_stdout").getvalue();
  const stderr: string = pyodide.globals.get("_stderr").getvalue();

  if (!succeeded) {
    // Include any outputs collected before the raise.
    return {
      raised: true,
      stdout,
      stderr: stderr || (exception ? exception.message : "unknown error"),
      outputs: collectOutputs(pyodide, options.resultOnly, false),
      produced: listProducedFiles(pyodide),
    };
  }

  return {
    raised: false,
    stdout,
    stderr,
    outputs: collectOutputs(pyodide, options.resultOnly),
    produced: listProducedFiles(pyodide),
  };
}

// Gather outputs: explicit display() calls first; fallback to result or the last open figure.
// get_fignums() not gcf() — gcf creates a blank figure when there is none.
function collectOutputs(pyodide: any, resultOnly = false, succeeded = true): PythonOutput[] {
  let json: string;
  try {
    pyodide.runPython(`
if ${resultOnly ? "True" : "False"}:
    for _previous_output in _nimbus_outputs:
        if _previous_output.get("key") is not None:
            globals().pop(_previous_output["key"], None)
    globals().pop("_previous_output", None)
    _nimbus_outputs = []
    _nimbus_output_bytes = 0
    _fallback = globals().get("result", None) if ${succeeded ? "True" : "False"} else None
    if _fallback is not None:
        display(_fallback)
elif not _nimbus_outputs:
    _fallback = globals().get("result", None) if _nimbus_result_assigned else None
    if _fallback is None and "matplotlib.pyplot" in sys.modules:
        _plt = sys.modules["matplotlib.pyplot"]
        _open_figures = _plt.get_fignums()
        if _open_figures:
            _fallback = _plt.figure(_open_figures[-1])
    if _fallback is not None:
        display(_fallback)
globals().pop("_fallback", None)
# Figures are captured as PNG by now, so drop the live ones. pyplot's registry
# is global and would otherwise grow for the life of the session.
if "matplotlib.pyplot" in sys.modules:
    sys.modules["matplotlib.pyplot"].close("all")
_nimbus_outputs_json = _json.dumps(_nimbus_outputs)
`);
    json = pyodide.globals.get("_nimbus_outputs_json") as string;
    // JS now owns the transport string. Release Python's JSON and its list of
    // base64/HTML payloads immediately so repeated plots do not retain every
    // result twice in the wasm heap. ndarray buffers stay in keyed globals
    // until the validated transfer loop consumes them.
    try {
      pyodide.runPython(`
globals().pop("_nimbus_outputs_json", None)
_nimbus_outputs = []
_nimbus_output_bytes = 0
`);
    } catch {
      // Validation below remains authoritative if cleanup cannot run.
    }
  } catch (error) {
    // Don't discard stdout/traceback — report capture failure as one more output.
    return [{
      kind: "text",
      label: "display error",
      text: error instanceof Error ? error.message : String(error),
    }];
  }

  let entries: unknown;
  try {
    entries = JSON.parse(json) as unknown;
  } catch {
    return [{ kind: "text", label: null, text: "Output payload could not be decoded." }];
  }
  if (!Array.isArray(entries) || entries.length > MAX_OUTPUTS_PER_RUN) {
    return [{
      kind: "text",
      label: "display refused",
      text: "Output payload did not match the bounded display-list contract.",
    }];
  }

  const outputs: PythonOutput[] = [];
  let transferredBytes = 0;
  for (const entry of entries) {
    if (entry == null || typeof entry !== "object" || Array.isArray(entry)) {
      outputs.push({
        kind: "text",
        label: "display refused",
        text: "Output payload contained an invalid display entry.",
      });
      continue;
    }
    const label = typeof entry.label === "string" ? entry.label : null;
    const kind = String(entry.kind ?? "text");
    const declaredBytes = typeof entry.bytes === "number" && Number.isSafeInteger(entry.bytes)
      ? entry.bytes
      : -1;
    const itemLimit = kind === "animation"
      ? MAX_ANIMATION_OUTPUT_BYTES
      : kind === "html"
        ? MAX_HTML_OUTPUT_BYTES
        : MAX_OUTPUT_ITEM_BYTES;
    if (declaredBytes < 0 || declaredBytes > itemLimit
        || transferredBytes + declaredBytes > MAX_OUTPUT_TOTAL_BYTES) {
      if (kind === "ndarray" && typeof entry.key === "string") {
        pyodide.runPython(`globals().pop(${JSON.stringify(entry.key)}, None); None`);
      }
      outputs.push({
        kind: "text",
        label: "display refused",
        text: declaredBytes < 0
          ? "Output payload had no trustworthy byte count."
          : `Output payload (${Math.ceil(declaredBytes / (1024 * 1024))} MiB) exceeded its transfer budget.`,
      });
      continue;
    }
    const stringBytes = (value: unknown) => new TextEncoder().encode(String(value ?? "")).byteLength;
    const actualStructuredBytes = kind === "image"
      ? String(entry.png ?? "").length + stringBytes(entry.svg)
      : kind === "html"
        ? stringBytes(entry.html)
        : kind === "animation"
          ? Array.isArray(entry.frames)
            ? entry.frames.reduce(
              (bytes: number, frame: unknown) => bytes + String(frame).length,
              0,
            )
            : -1
          : kind === "ndarray"
            ? declaredBytes
            : stringBytes(entry.text);
    const animationContractValid = kind !== "animation" || (
      Array.isArray(entry.frames)
      && entry.frames.length > 0
      && entry.frames.length <= MAX_ANIMATION_FRAMES
      && typeof entry.fps === "number"
      && Number.isInteger(entry.fps)
      && entry.fps >= 1
      && entry.fps <= 60
    );
    if (actualStructuredBytes !== declaredBytes || !animationContractValid) {
      if (kind === "ndarray" && typeof entry.key === "string") {
        pyodide.runPython(`globals().pop(${JSON.stringify(entry.key)}, None); None`);
      }
      outputs.push({
        kind: "text",
        label: "display refused",
        text: "Output payload metadata did not match its encoded content.",
      });
      continue;
    }
    transferredBytes += declaredBytes;
    switch (entry.kind) {
      case "ndarray": {
        const key = String(entry.key);
        const proxy = pyodide.globals.get(key);
        if (!proxy) {
          outputs.push({
            kind: "text",
            label: "display refused",
            text: "Displayed array buffer was no longer available.",
          });
          break;
        }
        // Copy raw bytes before releasing the wasm view; typed decoding preserves integer precision.
        let data: PythonNumericArray | null = null;
        let invalidArray = false;
        let view: any = null;
        try {
          view = proxy.getBuffer("u8");
          const rawShape = Array.isArray(entry.shape) ? entry.shape : null;
          const validShape = rawShape != null
            && rawShape.length <= 64
            && rawShape.every((dimension: unknown) =>
              typeof dimension === "number"
              && Number.isSafeInteger(dimension)
              && dimension >= 0)
            && (rawShape as number[]).reduce(
              (product: number, dimension: number) => product * dimension,
              1,
            )
              === declaredBytes / pythonDtypeBytes(String(entry.dtype));
          if ((view.data as Float32Array).byteLength !== declaredBytes || !validShape) {
            invalidArray = true;
          } else {
            data = decodePythonNumericArray(new Uint8Array(view.data as Uint8Array), String(entry.dtype));
          }
        } catch {
          invalidArray = true;
        } finally {
          view?.release();
          proxy.destroy();
          pyodide.runPython(`globals().pop(${JSON.stringify(key)}, None); None`);
        }
        if (invalidArray || data == null) {
          outputs.push({
            kind: "text",
            label: "display refused",
            text: "Displayed array shape or byte metadata did not match its buffer.",
          });
          break;
        }
        outputs.push({
          kind: "ndarray",
          label,
          data,
          shape: entry.shape as number[],
          dtype: String(entry.dtype),
        });
        break;
      }
      case "image":
        outputs.push({ kind: "image", label, pngBase64: String(entry.png), svg: typeof entry.svg === "string" ? entry.svg : undefined });
        break;
      case "html":
        outputs.push({
          kind: "html",
          label,
          html: String(entry.html),
          scripts: entry.scripts === true,
        });
        break;
      case "animation":
        outputs.push({
          kind: "animation",
          label,
          framesBase64: Array.isArray(entry.frames) ? entry.frames.map(String) : [],
          fps: typeof entry.fps === "number" ? entry.fps : 10,
        });
        break;
      default:
        outputs.push({ kind: "text", label, text: String(entry.text ?? "") });
    }
  }
  return outputs;
}

/** Files the cell wrote into the export directory, newest run only. */
function listProducedFiles(pyodide: any): ProducedFile[] {
  try {
    const names = pyodide.FS.readdir(PYODIDE_OUTPUT_DIR) as string[];
    return names
      .filter((name) => name !== "." && name !== "..")
      .slice(0, MAX_PRODUCED_FILES)
      .flatMap((name) => {
        const virtualPath = `${PYODIDE_OUTPUT_DIR}/${name}`;
        const { size, mode } = pyodide.FS.stat(virtualPath);
        return pyodide.FS.isFile(mode)
          ? [{ name, virtualPath, byteLength: size as number }]
          : [];
      })
      .filter((file) => Number.isFinite(file.byteLength) && file.byteLength >= 0);
  } catch {
    // The directory only exists once a cell has run.
    return [];
  }
}

/** Read one produced file's bytes for export to the real filesystem. */
export function readProducedFile(handle: PyodideHandle, file: ProducedFile): Uint8Array {
  const expectedPath = `${PYODIDE_OUTPUT_DIR}/${file.name}`;
  if (!file.name || file.name.includes("/") || file.name.includes("\\")
      || file.virtualPath !== expectedPath) {
    throw new Error("Produced-file export path is outside the Python output directory");
  }
  const stat = handle.pyodide.FS.stat(expectedPath);
  if (!handle.pyodide.FS.isFile(stat.mode)) {
    throw new Error(`'${file.name}' is not a regular produced file`);
  }
  if (!Number.isSafeInteger(stat.size) || stat.size < 0 || stat.size > MAX_PRODUCED_FILE_BYTES) {
    throw new Error(
      `'${file.name}' is too large to export in one transfer; the limit is 256 MiB`,
    );
  }
  if (stat.size !== file.byteLength) {
    throw new Error(`'${file.name}' changed after the cell completed; run the cell again`);
  }
  return handle.pyodide.FS.readFile(file.virtualPath) as Uint8Array;
}

/* ---------- Staged files ---------- */

// File staging (interpreter half). Budget + host read live on the main thread (Tauri IPC).
// Only virtual-FS writes belong here; they take bytes, not paths.

/** Create the staging directory, tolerating a second call. */
function ensureStageDir(pyodide: any, dir: string): void {
  try {
    pyodide.FS.mkdir(dir);
  } catch {
    // EEXIST after first call; directory existing is the desired state.
  }
}

/** Write already-read bytes into the interpreter's virtual filesystem. */
export function writeStagedFile(
  handle: PyodideHandle,
  directory: string,
  virtualPath: string,
  bytes: Uint8Array,
): void {
  ensureStageDir(handle.pyodide, directory);
  handle.pyodide.FS.writeFile(virtualPath, bytes);
}

/** Remove a staged file from the interpreter, freeing its heap. */
export function removeStagedFile(handle: PyodideHandle, virtualPath: string): void {
  try {
    handle.pyodide.FS.unlink(virtualPath);
  } catch {
    // Already gone (user deleted from Python); caller drops it from staged set either way.
  }
}

/** Expose staged files to Python as `files` dict. Rebuilt on every bind so released files disappear. */
export function bindStagedFiles(
  handle: PyodideHandle,
  staged: ReadonlyArray<{ name: string; virtualPath: string }>,
): void {
  handle.pyodide.globals.set(
    "_staged_files",
    Object.fromEntries(staged.map((file) => [file.name, file.virtualPath])),
  );
  handle.pyodide.runPython(
    'files = _staged_files.to_py() if hasattr(_staged_files, "to_py") else dict(_staged_files)',
  );
}
