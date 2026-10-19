/*
 * Wire contract between the UI thread and the interpreter worker.
 * Types and plain data only — safe to import from both sides without pulling in Pyodide or Tauri.
 * Traffic is bidirectional: load_field travels worker→main (only the main thread can read frames).
 */

import type { PythonNumericArray } from "./pythonNumericArray";
import type { NpyN } from "./npy";
import type { CompletionCandidate } from "@/lib/studio/pythonCompletion";
import type { GridInfo } from "@/lib/run/runOutputs";

/** Where user code writes files it wants to export. */
export const PYODIDE_OUTPUT_DIR = "/nimbus/out";

// Per-run output caps. An animation is base64 PNG per frame; unbounded = interpreter OOM.
export const MAX_ANIMATION_FRAMES = 600;
export const MAX_OUTPUTS_PER_RUN = 64;
/** Structured-clone/display ceilings. These cap the second copy created when
 * an interpreter result crosses into the UI, independently of field loading. */
export const MAX_OUTPUT_ITEM_BYTES = 64 * 1024 * 1024;
export const MAX_OUTPUT_TOTAL_BYTES = 256 * 1024 * 1024;
export const MAX_HTML_OUTPUT_BYTES = 8 * 1024 * 1024;
export const MAX_ANIMATION_OUTPUT_BYTES = 128 * 1024 * 1024;
/** Final stdout/stderr retained per stream and cell. Live output has a smaller
 * UI-tail cap; this protects the interpreter-side StringIO equivalent. */
export const MAX_CAPTURED_STREAM_CHARS = 2 * 1024 * 1024;
/** Produced files remain in the wasm filesystem. Export currently crosses the
 * worker boundary as one byte array, so it needs the same hard ceiling as an
 * imported data file until streaming export exists. */
export const MAX_PRODUCED_FILE_BYTES = 256 * 1024 * 1024;
export const MAX_PRODUCED_FILES = 1024;

/** dir() on a module can be very large; cap at what the popup can usefully show. */
export const MAX_COMPLETIONS = 200;

// Cap for load_series: wasm heap overflow aborts the interpreter rather than raising.
// np.stack peaks at ~2x this. Matched to the per-file staging budget (same 32-bit question).
export const MAX_SERIES_BYTES = 256 * 1024 * 1024;
/** Combined payload retained by load_dataset(). Individual field limits are
 * insufficient because several legal fields can still exhaust wasm together. */
export const MAX_DATASET_BYTES = 256 * 1024 * 1024;

// Theoretical wasm32 ceiling; browsers abort well below 4 GiB. Used as denominator for heap display.
export const WASM32_ADDRESS_CEILING_BYTES = 4 * 1024 * 1024 * 1024;

/** Status callback for the UI to render "Loading numpy…" etc. */
export type PyodideStatus =
  | { kind: "idle" }
  | { kind: "loading-runtime" }
  | { kind: "loading-package"; name: string }
  | { kind: "ready" }
  | { kind: "error"; message: string };

export type PyodideStatusListener = (s: PyodideStatus) => void;

export interface FieldBinding {
  name: string;
  field: NpyN;
}

/**
 * Everything a cell needs bound before it runs. `loadFrameField` is absent
 * (functions can't be cloned); the worker requests frames over the reverse channel.
 */
export interface InjectionContext {
  fields: FieldBinding[];
  grid: GridInfo | null;
  simulationTimeS: number | null;
  /** Field name → unit string from the step manifest, attached to the
   *  xarray variables so a plot can label axes without hardcoding. */
  fieldUnits?: Readonly<Record<string, string>>;
  /** Index of the frame whose fields are bound as globals. */
  frameIndex?: number | null;
  modelLevelIndices?: readonly number[];
  /** Simulator/publication step identifier. This is provenance, not an ordinal
   * into `steps`; Python exposes it separately as `step_index`. */
  stepIndex?: number;
  /** How many frames the active dataset has, so Python can iterate them. */
  frameCount?: number;
  /** Simulation time of every frame, for labelling a time series without
   *  loading each frame just to read its clock. */
  frameTimesS?: readonly (number | null)[];
  /**
   * Datasets reachable from user code, keyed by role ("primary",
   * "comparison"). Frame counts are included so a script can iterate a source
   * without loading a field just to discover its length.
   */
  sources?: ReadonlyArray<{ role: string; displayName: string; frameCount: number }>;
  /**
   * Load one field from an arbitrary frame of an arbitrary source. Supplied by
   * the widget because the run/dataset reader lives above this layer; without
   * it `load_field` is not defined at all rather than defined and broken.
   */
  loadFrameField?: (
    fieldName: string,
    frameIndex: number,
    source: string,
    window?: FieldWindow,
  ) => Promise<NpyN>;
}

/**
 * Reduction applied at the reader (main thread) before the field is copied into
 * the wasm heap. Stride/crop mean the interpreter only receives what was asked for.
 */
export interface FieldWindow {
  /** Take every nth cell per axis. All ones is the whole field. */
  readonly stride: readonly number[];
  /**
   * Half-open index bounds per axis, or null for the full extent. Applied
   * before the stride, so a crop and a subsample compose the obvious way.
   */
  readonly box: readonly (readonly [number, number])[] | null;
  /** Intersect the request with the simulator-published halo-free cell range. */
  readonly validOnly?: boolean;
  /** Physical-volume aggregate terrain blocks, then conservatively remap them
   * onto a separable physical-z lattice before copying into Python. */
  readonly terrainRemap?: boolean;
}

/** The half of an InjectionContext that can cross a postMessage boundary. */
export type SerializableInjectionContext =
  Omit<InjectionContext, "loadFrameField"> & {
    /** Whether the client will answer `loadFrameField` requests. Python defines
     *  `load_field` only when it can actually be served. */
    readonly canLoadFrames: boolean;
  };

/** One rendered result. Cells produce a list so multiple display() calls all land. */
export type PythonOutput =
  | { kind: "ndarray"; label: string | null; data: PythonNumericArray; shape: number[]; dtype?: string }
  | { kind: "image"; label: string | null; pngBase64: string; svg?: string }
  /**
   * Rendered inside a sandboxed iframe. `scripts` is set only for payloads that
   * genuinely need them (a matplotlib animation player); an object's
   * `_repr_html_` is markup and gets no script privileges.
   */
  | { kind: "html"; label: string | null; html: string; scripts: boolean }
  | { kind: "animation"; label: string | null; framesBase64: string[]; fps: number }
  | { kind: "text"; label: string | null; text: string };

/** A file the cell wrote into the interpreter's output directory. */
export interface ProducedFile {
  readonly name: string;
  readonly virtualPath: string;
  readonly byteLength: number;
}

export interface RunResult {
  /** True on an exception. Distinct from non-empty stderr (DeprecationWarnings don't raise). */
  raised: boolean;
  /** Captured stdout (print() output). */
  stdout: string;
  /** Captured stderr (warnings + tracebacks). */
  stderr: string;
  /** Everything the cell asked to show, in the order it asked. */
  outputs: PythonOutput[];
  /** Files the cell wrote, offered for export to the real filesystem. */
  produced: ProducedFile[];
}

export interface RunOptions {
  readonly resultOnly?: boolean;
  /** Per-chunk streaming callback; final stdout/stderr on the result is still authoritative. */
  readonly onStream?: (stream: "stdout" | "stderr", text: string) => void;
  /**
   * Whether to empty the export directory first. Default true.
   * Set false for a mid-document fragment so earlier cells' outputs survive.
   */
  readonly resetProducedFiles?: boolean;
}

/** What a run needs that survives cloning; `onStream` is served by an event. */
export interface SerializableRunOptions {
  readonly resultOnly?: boolean;
  readonly streaming: boolean;
  readonly resetProducedFiles?: boolean;
}

/** One name bound in the interpreter's global namespace. */
export interface NamespaceEntry {
  readonly name: string;
  readonly kind: string;
  readonly detail: string | null;
}

/* ---------- Envelopes ---------- */

// Error flattened for postMessage transport (structured clone drops subclasses).
// Client rebuilds the correct subclass from name + extra fields.
export interface SerializedError {
  readonly name: string;
  readonly message: string;
  /** MissingPackageError only. */
  readonly pkg?: string;
  readonly underlying?: string;
  /** StagingBudgetError only. */
  readonly hostPath?: string;
  readonly byteLength?: number;
  /** Interpreter is gone; nothing can run until the worker is replaced. */
  readonly fatal?: boolean;
}

export interface RpcRequest {
  readonly kind: "request";
  readonly id: number;
  readonly op: string;
  readonly args: readonly unknown[];
}

export type RpcResponse =
  | { readonly kind: "response"; readonly id: number; readonly ok: true; readonly value: unknown }
  | { readonly kind: "response"; readonly id: number; readonly ok: false; readonly error: SerializedError };

/** Unsolicited worker → main traffic: progress that has no reply. */
export type RpcEvent =
  | { readonly kind: "event"; readonly event: "status"; readonly status: PyodideStatus }
  | {
      readonly kind: "event";
      readonly event: "stream";
      readonly stream: "stdout" | "stderr";
      readonly text: string;
    };

export type RpcMessage = RpcRequest | RpcResponse | RpcEvent;

/** Ops the worker serves. */
export const WORKER_OPS = {
  boot: "boot",
  bindContext: "bindContext",
  runCell: "runCell",
  loadPackage: "loadPackage",
  micropipInstall: "micropipInstall",
  installPackage: "installPackage",
  loadedPackages: "loadedPackages",
  complete: "complete",
  inspectNamespace: "inspectNamespace",
  heapBytes: "heapBytes",
  readProducedFile: "readProducedFile",
  writeStagedFile: "writeStagedFile",
  removeStagedFile: "removeStagedFile",
  bindStagedFiles: "bindStagedFiles",
} as const;

/** Ops the main thread serves for the worker. */
export const CLIENT_OPS = {
  loadFrameField: "loadFrameField",
} as const;

export type { CompletionCandidate, NpyN };
