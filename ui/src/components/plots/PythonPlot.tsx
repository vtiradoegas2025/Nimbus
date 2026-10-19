import { pythonArrayDtype, pythonArrayPreview, type PythonNumericArray } from "@/lib/platform/pythonNumericArray";
import {
  lazy,
  memo,
  Suspense,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
  type KeyboardEvent as ReactKeyboardEvent,
  type SetStateAction,
} from "react";
import type { Data, Layout } from "plotly.js";
import {
  readFieldAny,
  readFieldAnyWindows,
  registeredFieldRank,
} from "@/lib/run/runOutputs";
import {
  createSubsampledWindowAssembly,
  planPythonDenseFieldTiles,
  planPythonFieldTiles,
  resolveWindow,
} from "@/lib/run/fieldWindow";
import type {
  FieldWindowGeometry,
  Npy3D,
  NpyN,
  PythonFieldMetadata,
} from "@/lib/platform/npy";
import {
  MAX_OUTPUT_TOTAL_BYTES,
  MAX_PRODUCED_FILE_BYTES,
  type FieldWindow,
} from "@/lib/platform/pyodideProtocol";
import type { StepInfo } from "@/lib/run/runOutputs";
import {
  fieldSupportsBoundedWindows,
  fieldWindowAggregation,
} from "@/lib/plot/fieldLoadPlanning";
import {
  detectMissingModule,
  getPythonRuntimeClient,
  isBundledPyodideImport,
  MissingPackageError,
  packageNameForImport,
  PyodideFatalError,
  PythonStoppedError,
  type NamespaceEntry,
  type ProducedFile,
  type PyodideStatus,
  type PythonOutput,
  type RunResult,
} from "@/lib/platform/pyodideRuntime";
import {
  base64ToBytes,
  encodeFramesToGif,
  MAX_CSV_CELLS,
} from "@/lib/platform/pythonExport";
import { pythonArrayExportService } from "@/lib/platform/pythonArrayExportClient";
import { PYODIDE_CATALOG } from "@/lib/platform/pyodidePackages";
import { offsetForLine, splitTraceback } from "@/lib/studio/pythonTraceback";
import {
  decideStop,
  STOP_STATUS_INTERRUPTED,
  stopButtonLabel,
  stopButtonTitle,
} from "@/lib/studio/pythonStop";
import {
  budgetExhausted,
  detectReferencedFields,
  fitsInBudget,
  NO_SKIPPED_FIELDS,
  MAX_AUTO_INJECT_BYTES,
  skippedFieldsNotice,
  type SkippedField,
} from "@/lib/studio/pythonFieldRefs";
import { isVolumetricField } from "@/lib/plot/fieldProvenance";
import { preparePythonArrayDisplay, pythonArrayViewportFromRelayout } from "@/lib/plot/pythonArrayDisplay";
import { pythonFieldCoordinates } from "@/lib/plot/pythonFieldCoordinates";
import { terrainWindowToCartesianField } from "@/lib/science/cartesianInterp";
import {
  patchSession,
  readSession,
  sessionKey,
  subscribeSession,
  type CellRun,
  type LiveStream,
  type PythonRunSession,
  type ResultFrame,
} from "@/lib/studio/pythonSession";
import {
  cellIndexAtOffset,
  cellLabel,
  cellSourceForExecution,
  isCellEmpty,
  splitPythonCells,
  type PythonCell,
} from "@/lib/studio/pythonCells";
import {
  activateBuffer,
  activeBuffer,
  addScratchBuffer,
  canAddBuffer,
  closeBuffer,
  createWorkspace,
  fileName as fileNameOf,
  isBufferDirty,
  WorkspaceOperationOwner,
  LEGACY_SNIPPET_STORAGE_KEY,
  markBufferLoaded,
  markBufferSaved,
  MAX_BUFFERS,
  migrateLegacySnippet,
  openPathInWorkspace,
  parseWorkspace,
  pythonWorkspaceSeedKey,
  pythonWorkspaceStorageKey,
  seedProjectWorkspace,
  serializeWorkspace,
  setBufferCode,
  workspaceRecoveryWarning,
  type PythonBuffer,
  type PythonWorkspace,
} from "@/lib/studio/pythonWorkspace";
import {
  getPluginsDir,
  listPlugins,
  openPath,
  pickDataFiles,
  pickExportSavePath,
  pickPySnippetLoadPath,
  pickPySnippetSavePath,
  readPySnippet,
  writeBinaryFile,
  writePySnippet,
  type PluginEntry,
} from "@/lib/platform/tauri";
import {
  bindStagedFiles,
  formatBytes,
  MAX_STAGED_TOTAL_BYTES,
  restageFiles,
  stageFile,
  unstageFile,
  type StagedFile,
} from "@/lib/platform/pyodideFiles";
import { useResizableWidth } from "@/lib/core/hooks";
import { Icon } from "@/components/shell/Icon";
import { ResizeHandle } from "@/components/shell/ResizeHandle";
import { PlotCanvas } from "@/components/studio/PlotCanvas";
import { usePlaneViewport, type PlotComponentProps, type PlotDatasetHandle } from "./common";
import type { PlaneViewportBounds } from "@/lib/plot/viewSync";
import { usePaneContext } from "./paneContext";
import { useOutputHierarchySource } from "@/lib/run/useOutputHierarchySource";
import { OutputHierarchyControls } from "./OutputHierarchyControls";
import { PlotCompressionControls } from "./PlotSourceDisclosure";
import "./PythonPlot.css";

// CodeMirror (~80 KB). Lazy-load so non-Python plot users don't pay for it.
// Type-only handle import carries no runtime dependency and keeps the chunk split.
import type { CodeEditorHandle } from "@/components/studio/CodeEditor";

const CodeEditor = lazy(() =>
  import("@/components/studio/CodeEditor").then((m) => ({ default: m.CodeEditor })),
);

// float32 value plus float64 x/y/z physical centroid and exact cell volume.
const PYTHON_TERRAIN_BYTES_PER_CELL = 36;

function pythonReadWindow(
  shape: readonly number[],
  requested: FieldWindow | undefined,
  step: StepInfo,
): FieldWindow {
  if (requested?.validOnly === false || !step.grid?.validIndexRange) {
    return requested ?? { stride: [], box: null };
  }
  const lower = step.grid.validIndexRange.lower;
  const upper = step.grid.validIndexRange.upper;
  return {
    stride: requested?.stride ?? [],
    validOnly: true,
    terrainRemap: requested?.terrainRemap,
    box: shape.map((count, axis) => {
      const requestedBounds = requested?.box?.[axis];
      const start = Math.max(0, lower[axis] ?? 0, requestedBounds?.[0] ?? 0);
      const stop = Math.min(count, (upper[axis] ?? count - 1) + 1, requestedBounds?.[1] ?? count);
      return [start, Math.max(start, stop)] as const;
    }),
  };
}

function remapPythonTerrainField(
  field: NpyN,
  step: StepInfo,
  requested: boolean | undefined,
): { field: NpyN; metadataStep: StepInfo; terrainRemapped: boolean } {
  if (!requested || !step.grid?.terrainFollowing) {
    return { field, metadataStep: step, terrainRemapped: false };
  }
  if (field.shape.length !== 3) {
    throw new Error("terrain_remap=True requires a rank-3 field");
  }
  const physical = terrainWindowToCartesianField(field as Npy3D, step.grid);
  const lower = physical.origin.map(
    (origin, axis) => origin - 0.5 * physical.spacing[axis],
  ) as [number, number, number];
  const upper = physical.origin.map(
    (origin, axis) => origin + (physical.field.shape[axis] - 0.5) * physical.spacing[axis],
  ) as [number, number, number];
  return {
    field: physical.field,
    metadataStep: {
      ...step,
      grid: {
        ...step.grid,
        nr: physical.field.shape[0],
        nth: physical.field.shape[1],
        nz: physical.field.shape[2],
        terrainFollowing: false,
        geometryKind: "physical",
        drM: physical.spacing[0],
        dyM: physical.spacing[1],
        dzM: physical.spacing[2],
        scalarOriginM: physical.origin,
        scalarCoordinatesM: physical.scalarCoordinatesM,
        physicalBoundsM: { lower, upper },
        validIndexRange: {
          lower: [0, 0, 0],
          upper: physical.field.shape.map((count) => count - 1) as [number, number, number],
        },
        haloWidth: { lower: [0, 0, 0], upper: [0, 0, 0] },
      },
    },
    terrainRemapped: true,
  };
}

function pythonFieldMetadata(
  field: NpyN,
  sourceShape: readonly number[],
  window: FieldWindow,
  step: StepInfo,
  frameIndex: number,
  fieldName: string,
  source: PlotDatasetHandle,
  terrainRemapped = false,
  fidelity?: {
    sourceShape: readonly number[];
    window: FieldWindow;
    geometry?: FieldWindowGeometry;
  },
): PythonFieldMetadata {
  const grid = step.grid;
  const resolved = resolveWindow(sourceShape, window);
  const coordinates = pythonFieldCoordinates(grid, sourceShape, field.shape, resolved.starts, resolved.steps);
  const fidelityShape = fidelity?.sourceShape ?? sourceShape;
  const fidelityWindow = fidelity?.window ?? window;
  const fidelityResolved = resolveWindow(fidelityShape, fidelityWindow);
  const geometry = fidelity?.geometry ?? field.windowGeometry;
  const nativeResolution = fidelityResolved.steps.every((step) => step === 1);
  return {
    fieldName,
    unit: step.fieldUnits[fieldName] ?? null,
    source: source.role,
    sourceName: source.displayName,
    frameIndex,
    stepIndex: step.index,
    simulationTimeS: step.simulationTimeS,
    coordinateSystem: grid?.coordinateSystem ?? "unknown",
    terrainFollowing: grid?.terrainFollowing === true,
    sourceTerrainFollowing: terrainRemapped || grid?.terrainFollowing === true,
    coordinateResampling: terrainRemapped ? "conservative-terrain-to-cartesian" : "none",
    validOnly: window.validOnly === true,
    dtype: "float32",
    sourceShape: [...fidelityShape],
    outputShape: [...field.shape],
    sampling: nativeResolution ? "native" : geometry?.aggregation ?? "point_sample",
    aggregation: geometry?.aggregation ?? null,
    weighting: geometry?.weighting ?? null,
    sampleLocation: geometry?.sampleLocation ?? null,
    sourceFormat: geometry?.sourceFormat ?? null,
    compression: step.fieldCompression?.[fieldName] ?? null,
    stride: [...fidelityResolved.steps],
    box: fidelityResolved.starts.map(
      (start, axis) => [start, fidelityResolved.stops[axis]] as [number, number],
    ),
    coordinates,
  };
}

/** Exact registry metadata wins. A directly mounted/embed plot may receive
 * already-qualified StepInfo without listSteps() having populated the global
 * registry, so its own published catalog remains a trustworthy rank fallback. */
function pythonFieldRank(
  runDir: string,
  step: StepInfo,
  fieldName: string,
): number | null {
  const registered = registeredFieldRank(
    runDir,
    step.index,
    fieldName,
    step.publicationGeneration,
  );
  if (registered != null) return registered;
  if (!step.fields.includes(fieldName)) return null;
  return isVolumetricField(fieldName) ? 3 : 2;
}

function needsDensePythonFallback(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return message.includes("ZFP field-window reconstruction needs")
    || message.includes("terrain field window needs");
}

function pythonOutputRetainedBytes(output: PythonOutput): number {
  switch (output.kind) {
    case "ndarray": return output.data.byteLength;
    case "image": return (output.pngBase64.length + (output.svg?.length ?? 0)) * 2;
    case "html": return output.html.length * 2;
    case "animation": return output.framesBase64.reduce(
      (bytes, frame) => bytes + frame.length * 2,
      0,
    );
    case "text": return output.text.length * 2;
  }
}

function retainRunResult(
  result: RunResult,
  byteBudget: number,
): { result: RunResult; bytes: number } {
  let remaining = Math.max(0, byteBudget);
  let retainedBytes = 0;
  let dropped = false;
  const takeText = (value: string): string => {
    const bytes = value.length * 2;
    if (bytes <= remaining) {
      remaining -= bytes;
      retainedBytes += bytes;
      return value;
    }
    dropped ||= value.length > 0;
    const marker = "\n[Nimbus: Run result retention limit reached]\n";
    const characters = Math.max(0, Math.floor(remaining / 2) - marker.length);
    const kept = characters > 0 ? value.slice(-characters) : "";
    const captured = kept + (remaining >= marker.length * 2 ? marker : "");
    const capturedBytes = captured.length * 2;
    remaining -= capturedBytes;
    retainedBytes += capturedBytes;
    return captured;
  };
  // Preserve traceback/warning tails before ordinary stdout when the Run is at
  // its ceiling; the final exception is the part needed to diagnose the cell.
  const stderr = takeText(result.stderr);
  const stdout = takeText(result.stdout);
  const outputs: PythonOutput[] = [];
  for (const output of result.outputs) {
    const bytes = pythonOutputRetainedBytes(output);
    if (bytes > remaining) {
      dropped = true;
      continue;
    }
    outputs.push(output);
    remaining -= bytes;
    retainedBytes += bytes;
  }
  if (dropped) {
    const notice: PythonOutput = {
      kind: "text",
      label: "display refused",
      text: "Additional cell output was discarded after this Run reached its 256 MiB retained-result limit.",
    };
    const bytes = pythonOutputRetainedBytes(notice);
    if (bytes <= remaining) {
      outputs.push(notice);
      retainedBytes += bytes;
    }
  }
  return { result: dropped ? { ...result, stdout, stderr, outputs } : result, bytes: retainedBytes };
}

/* Python plot tab — Pyodide REPL. Field names in user code are scanned and
 * the matching arrays are injected before exec; grid metadata always available.
 * Cells produce a list of outputs via display()/result/matplotlib capture. */

const STARTER_CODE = `# Fidelity-first starter
#
# This draws one horizontal slice from the frame on screen. It never smooths or
# interpolates the field: every coloured rectangle is one returned model cell.
# TARGET_SAMPLES bounds browser/Matplotlib work on very large domains; set it to
# None to request every valid horizontal cell, or use load_field(..., box=...)
# for a native-resolution region of interest.
#
# Useful globals: nr/nth/nz, coordinate_units, x_coordinates/y_coordinates/z_coordinates,
# x_m/y_m/z_m (empty for non-metre axes), theta_rad, valid_lower/valid_upper,
# coordinate_system, terrain_following, simulation_time_s, frame_index.
# load_field() reads bounded/strided data before it crosses into Python, while
# field_metadata() reports the exact coordinates and sampling provenance.
# Other frames and datasets are explicit:
#   prior_w = await load_field("w", frame=max(0, frame_index - 1), stride=4)
#   compare_w = await load_field("w", source="comparison", stride=4)
# load_series() stacks selected frames. dataset() / load_dataset() return xarray
# objects with named coordinates and units. Staged local data is in files["cm1out.nc"].
# For terrain-aware analysis:
#   terrain_w = await load_field("w", stride=4, terrain_remap=terrain_following)
#   terrain_provenance = field_metadata("w")
# This performs conservative physical-volume reduction and physical-z remapping;
# inspect the returned coordinate and provenance metadata before interpretation.
#
# numpy and matplotlib are ready. Bundled on demand: xarray, pandas, scipy,
# netCDF4, h5py, cftime, cartopy, pyproj, shapely, sklearn, statsmodels, sympy,
# zarr, and PIL. display() accepts figures, arrays, tables, and text; if display()
# is omitted, the current figure is captured automatically.

# %% Select and load
import numpy as np
import matplotlib.pyplot as plt

TARGET_SAMPLES = 600
TARGET_HEIGHT_M = 1000.0
FIELD_CANDIDATES = ("w", "reflectivity_dbz", "theta_prime", "theta", "qv")

field = None
field_name = None
metadata = None
load_errors = []

have_grid = nr >= 2 and nth >= 2 and nz >= 1
if have_grid and "load_field" in globals():
    i0, j0, k0 = (int(value) for value in valid_lower)
    i1, j1, k1 = (int(value) + 1 for value in valid_upper)

    # Use the level nearest 1 km on a separable Cartesian grid. A terrain-following
    # frame uses its lowest valid model level and says so in the figure.
    if not terrain_following and len(z_m) >= k1:
        valid_z = np.asarray(z_m[k0:k1], dtype=np.float64)
        k = k0 + int(np.argmin(np.abs(valid_z - TARGET_HEIGHT_M)))
    else:
        k = k0

    valid_nx = max(1, i1 - i0)
    valid_ny = max(1, j1 - j0)
    if TARGET_SAMPLES is None:
        stride_x = stride_y = 1
    else:
        stride_x = max(1, int(np.ceil(valid_nx / TARGET_SAMPLES)))
        stride_y = max(1, int(np.ceil(valid_ny / TARGET_SAMPLES)))

    request_box = ((i0, i1), (j0, j1), (k, k + 1))
    request_stride = (stride_x, stride_y, 1)

    for candidate in FIELD_CANDIDATES:
        try:
            loaded = await load_field(
                candidate,
                stride=request_stride,
                box=request_box,
                terrain_remap=False,
            )
            if loaded.ndim != 3 or loaded.shape[2] != 1:
                raise ValueError("expected one rank-3 model level, got " + str(loaded.shape))
            field = loaded
            field_name = candidate
            metadata = field_metadata(candidate)
            break
        except Exception as error:
            load_errors.append(candidate + ": " + str(error))

# %% Draw exact returned cells
if field is None:
    # This verifies the Python/Matplotlib path without drawing a fictional storm.
    # The title and console make it impossible to confuse with simulation output.
    print("source: synthetic plotting check — NO SIMULATION FIELD WAS USED")
    if have_grid and load_errors:
        print("no starter field could be loaded:")
        for message in load_errors:
            print("  " + message)

    test_x = np.linspace(-1.0, 1.0, 80, dtype=np.float32)
    test_y = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    xx, yy = np.meshgrid(test_x, test_y, indexing="xy")
    test_pattern = np.sin(4.0 * xx) * np.cos(3.0 * yy) + 0.35 * xx

    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    image = ax.imshow(
        test_pattern,
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
        extent=(-1.0, 1.0, -1.0, 1.0),
        aspect="equal",
    )
    ax.set_title("Analytic plotting check — not simulation data", color="crimson")
    ax.set_xlabel("analytic x")
    ax.set_ylabel("analytic y")
    fig.colorbar(image, ax=ax, label="dimensionless test value")
    fig.tight_layout()
    display(fig, label="analytic plotting check (no simulation data)")
    plt.close(fig)
else:
    plane = np.asarray(field[:, :, 0], dtype=np.float32).T
    coordinates = metadata["coordinates"]
    axis_x = np.asarray(coordinates[0]["values"], dtype=np.float64)
    axis_y = np.asarray(coordinates[1]["values"], dtype=np.float64)
    axis_z = np.asarray(coordinates[2]["values"], dtype=np.float64)
    unit = metadata.get("unit") or "unknown unit"
    stride = tuple(int(value) for value in metadata["stride"])
    native = stride[0] == 1 and stride[1] == 1

    cylindrical = str(coordinate_system).lower().startswith("cyl")
    if cylindrical:
        plot_x = axis_x / 1000.0
        plot_y = np.degrees(axis_y)
        x_label = "radius (km)"
        y_label = "azimuth (degrees)"
    else:
        plot_x = axis_x / 1000.0
        plot_y = axis_y / 1000.0
        x_label = "x / east (km)"
        y_label = "y / north (km)"

    finite = plane[np.isfinite(plane)]
    if finite.size == 0:
        raise ValueError(field_name + " contains no finite values on the selected level")

    diverging = field_name in ("u", "v", "w", "theta_prime")
    if diverging:
        reach = float(np.nanpercentile(np.abs(finite), 99.0))
        reach = reach if reach > 0.0 else 1.0
        vmin, vmax, cmap = -reach, reach, "RdBu_r"
    else:
        vmin, vmax = (float(value) for value in np.nanpercentile(finite, (1.0, 99.0)))
        if not vmax > vmin:
            padding = max(1.0, abs(vmin) * 0.01)
            vmin, vmax = vmin - padding, vmax + padding
        cmap = "viridis"

    level_text = (
        "lowest valid model level"
        if terrain_following
        else "z = " + format(float(axis_z[0]) / 1000.0, ".2f") + " km"
    )
    sampling_text = (
        "native horizontal cells"
        if native
        else "point sampled every " + str(stride[0]) + " × " + str(stride[1])
    )

    print(
        "source: open frame · field", field_name,
        "· returned", tuple(int(value) for value in field.shape),
    )
    print("sampling:", sampling_text, "· nearest-cell display · no interpolation")

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    mesh = ax.pcolormesh(
        plot_x,
        plot_y,
        plane,
        shading="nearest",
        antialiased=False,
        rasterized=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if not cylindrical:
        ax.set_aspect("equal")
    ax.set_title(field_name + " · " + level_text)
    ax.text(
        0.01,
        0.01,
        sampling_text + " · nearest cells, no interpolation",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )
    fig.colorbar(mesh, ax=ax, label=field_name + " (" + unit + ")")
    fig.tight_layout()
    display(fig, label=field_name + " horizontal slice")
    plt.close(fig)
`;

// Recovery store: carries scratch text and unsaved edits so a crash or tab
// switch does not lose them. Canonical saved files live in ~/.nimbus/plugins.

// Whether the plugin gallery sidebar was open last session.
const GALLERY_OPEN_STORAGE_KEY = "nimbus-studio:python-gallery:open";

// Debounce localStorage writes to avoid slamming setItem on every keystroke.
const PERSIST_DEBOUNCE_MS = 500;

// 60 ms: one frame of coalescing so a chatty loop costs bounded renders.
const STREAM_FLUSH_MS = 60;

// Live preview keeps only the tail; the full text is delivered on completion.
const MAX_LIVE_STREAM_CHARS = 120_000;

// Editor width stored in px (not fraction) so resizing the outer pane does not
// silently reclaim space the user took from it.
const EDITOR_WIDTH_STORAGE_KEY = "nimbus-studio:layout:python-editor-width";
const EDITOR_WIDTH_DEFAULT = 460;
const EDITOR_WIDTH_MIN = 260;
const EDITOR_WIDTH_MAX = 1100;

/** useState-shaped setter for one field of the run session; writes to the
 *  module-level store so a finishing run can record its result after unmount. */
function useSessionSetter<K extends keyof PythonRunSession>(
  key: string,
  field: K,
): (value: SetStateAction<PythonRunSession[K]>) => void {
  return useCallback((value: SetStateAction<PythonRunSession[K]>) => {
    patchSession(key, (previous) => ({
      [field]: typeof value === "function"
        ? (value as (current: PythonRunSession[K]) => PythonRunSession[K])(previous[field])
        : value,
    }) as Partial<PythonRunSession>);
  }, [key, field]);
}

function appendCapped(existing: string, chunk: string): { text: string; truncated: boolean } {
  const combined = existing + chunk;
  if (combined.length <= MAX_LIVE_STREAM_CHARS) return { text: combined, truncated: false };
  return { text: combined.slice(combined.length - MAX_LIVE_STREAM_CHARS), truncated: true };
}

/** True when Save writes in-place. Also a type predicate so the write path
 *  can use buffer.path without a cast. A path-only (needsReload) buffer fails
 *  this so we always prompt rather than overwriting an unread file. */
function saveWritesInPlace(
  buffer: PythonBuffer,
): buffer is PythonBuffer & { path: string } {
  return buffer.path !== null && !buffer.needsReload;
}

interface LoadedWorkspace {
  readonly workspace: PythonWorkspace;
  /** True when this store did not exist and open files were carried into it. */
  readonly seeded: boolean;
}

function loadPersistedWorkspace(storageKey: string, seedKey: string | null): LoadedWorkspace {
  if (typeof window === "undefined") {
    return { workspace: createWorkspace(STARTER_CODE), seeded: false };
  }
  try {
    const raw = window.localStorage.getItem(storageKey);
    if (raw) return { workspace: parseWorkspace(raw, STARTER_CODE), seeded: false };
    if (seedKey !== null) {
      // First open: seed from the previous session's open files, not a blank buffer.
      const seeded = seedProjectWorkspace(
        window.localStorage.getItem(seedKey),
        STARTER_CODE,
      );
      return { workspace: seeded, seeded: seeded.buffers.some((entry) => entry.path !== null) };
    }
    // Migrate legacy single-buffer store on first upgrade.
    return {
      workspace: migrateLegacySnippet(
        window.localStorage.getItem(LEGACY_SNIPPET_STORAGE_KEY),
        STARTER_CODE,
      ),
      seeded: false,
    };
  } catch {
    // localStorage may be blocked by quota or sandbox; fall through.
  }
  return { workspace: createWorkspace(STARTER_CODE), seeded: false };
}

/** Write the recovery store; returns an error message on failure (e.g. quota
 *  exceeded) or null on success. Reports rather than swallows quota errors. */
function persistWorkspace(storageKey: string, workspace: PythonWorkspace): string | null {
  try {
    const serialized = serializeWorkspace(workspace);
    if (window.localStorage.getItem(storageKey) !== serialized) window.localStorage.setItem(storageKey, serialized);
    return null;
  } catch (e) {
    const detail = e instanceof Error ? e.message : String(e);
    return `Could not save recovery state (${detail}). Unsaved buffers will not survive a reload — save them to disk.`;
  }
}

export function PythonPlot({
  runDir: rootRunDir,
  steps: rootSteps,
  stepIdx: rootStepIdx,
  datasets,
  projectId = null,
}: PlotComponentProps) {
  const outputHierarchySource = useOutputHierarchySource({
    runDir: rootRunDir,
    steps: rootSteps,
    stepIdx: rootStepIdx,
    scope: "shared",
  });
  const runDir = outputHierarchySource.sourceRunDir;
  const steps = outputHierarchySource.sourceSteps;
  const stepIdx = outputHierarchySource.sourceStepIdx >= 0
    ? outputHierarchySource.sourceStepIdx
    : rootStepIdx;
  const currentStep = outputHierarchySource.currentStep;
  /* Falls back to a synthetic primary handle when caller omits datasets,
   * so load_field works in the harness and single-pane embeds. */
  const sourceHandles = useMemo<readonly PlotDatasetHandle[]>(
    () => (datasets && datasets.length > 0
      ? datasets.map((source) => source.role === "primary"
          ? { ...source, runDir, steps }
          : source)
      : [{ id: runDir, role: "primary", displayName: "primary", runDir, steps }]),
    [datasets, runDir, steps],
  );
  /* Recovery store key: scoped per project and per pane. Two Python panes
   * are independent sessions — a shared store caused each write to clobber
   * the other pane's buffers. */
  const { paneId } = usePaneContext();
  const storageKey = pythonWorkspaceStorageKey(projectId, paneId);
  const seedKey = pythonWorkspaceSeedKey(projectId, paneId);
  // Lazy init so we read localStorage exactly once on mount.
  const [workspace, setWorkspace] = useState<PythonWorkspace>(
    () => loadPersistedWorkspace(storageKey, seedKey).workspace,
  );
  /* Always write through this ref, not storageKey: the two disagree for the
   * one render between a project change and the buffer swap. */
  const loadedKeyRef = useRef(storageKey);
  const workspaceOperationRef = useRef(new WorkspaceOperationOwner(storageKey));
  workspaceOperationRef.current.select(storageKey);
  useEffect(() => () => workspaceOperationRef.current.retire(), []);
  /** Non-null when the recovery store could not be written; see persistWorkspace. */
  const [persistError, setPersistError] = useState<string | null>(null);
  const buffer = activeBuffer(workspace);
  const code = buffer.code;
  const recoveryWarning = useMemo(() => workspaceRecoveryWarning(workspace), [workspace]);
  // Shim: edits land on the active buffer; callers see a plain string.
  const setCode = useCallback((next: string) => {
    setWorkspace((current) => setBufferCode(current, current.activeId, next));
  }, []);
  const [pyStatus, setPyStatus] = useState<PyodideStatus>({ kind: "idle" });
  const [editorWidth, adjustEditorWidth] = useResizableWidth(
    EDITOR_WIDTH_STORAGE_KEY,
    EDITOR_WIDTH_DEFAULT,
    EDITOR_WIDTH_MIN,
    EDITOR_WIDTH_MAX,
  );
  /* Run state lives in a module store so it survives unmount; keyed by
   * project + pane, as two Python panes are independent sessions. */
  const runKey = sessionKey(projectId, paneId);
  // Every project/pane owns a worker and callback channel. Keeping the client
  // stable across route unmounts lets an in-flight run finish without sharing
  // namespace, files, output, Stop, or Restart with another Python pane.
  const runtime = useMemo(() => getPythonRuntimeClient(runKey), [runKey]);
  const runtimeCanInterrupt = runtime.canInterrupt();
  const completeNamesForRuntime = useCallback(
    (path: readonly string[], partial: string) => runtime.completeNames(path, partial),
    [runtime],
  );
  const session = useSyncExternalStore(
    useCallback((onChange: () => void) => subscribeSession(runKey, onChange), [runKey]),
    useCallback(() => readSession(runKey), [runKey]),
  );
  const {
    running,
    stopStage,
    completedRuns,
    outputBufferId,
    cellRuns,
    displayedCell,
    lastRunCellIndices,
    liveStream,
    skippedFields,
    ioStatus,
    loadErr,
    fatalError,
    stagedFiles,
  } = session;
  /* Session setters: useState shape preserved so call sites are unchanged;
   * writes go to the module store so a finishing run records after unmount. */
  const setStopStage = useSessionSetter(runKey, "stopStage");
  const setDisplayedCell = useSessionSetter(runKey, "displayedCell");
  const setIoStatus = useSessionSetter(runKey, "ioStatus");
  const arrayExportControllerRef = useRef<AbortController | null>(null);
  const [arrayExporting, setArrayExporting] = useState(false);
  useEffect(() => () => arrayExportControllerRef.current?.abort(), [runKey]);
  const setStagedFiles = useSessionSetter(runKey, "stagedFiles");
  /* `running` ref: the escalation timer's closure would capture a stale true;
   * read the ref instead to check if the cell is still in flight. */
  const runningRef = useRef(false);
  runningRef.current = running;
  /** The armed escalation, so a finished run and an unmount can both cancel it. */
  const stopTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [activePane, setActivePane] = useState<"result" | "output">("result");
  const outputId = useId();
  const editorId = useId();
  const bufferTabRefs = useRef(new Map<string, HTMLButtonElement>());
  const focusBufferAfterCloseRef = useRef(false);
  const [packagesOpen, setPackagesOpen] = useState(false);
  const [stagingBusy, setStagingBusy] = useState(false);
  const [stagingError, setStagingError] = useState<string | null>(null);
  const [reloadFailedIds, setReloadFailedIds] = useState<ReadonlySet<string>>(() => new Set());
  const stagedBytes = useMemo(
    () => stagedFiles.reduce((total, file) => total + file.byteLength, 0),
    [stagedFiles],
  );
  // Parsed from stderr when the user imports an unloaded package; drives the
  // banner's Load button.
  const [missingModule, setMissingModule] = useState<string | null>(null);
  const [packageLoading, setPackageLoading] = useState(false);
  const [packageError, setPackageError] = useState<string | null>(null);
  // Gates "Try micropip" — only set when loadPackage fails with MissingPackageError.
  const [canTryMicropip, setCanTryMicropip] = useState(false);
  const [loadedPackages, setLoadedPackages] = useState<ReadonlySet<string>>(() => new Set());
  // Which catalog row is installing (only that row shows a spinner).
  const [installingPackage, setInstallingPackage] = useState<string | null>(null);
  const [installQuery, setInstallQuery] = useState("");
  /* Namespace and heap are snapshots refreshed after each run, not live
   * subscriptions — polling would contend with user code on the interpreter thread. */
  const [namespace, setNamespace] = useState<readonly NamespaceEntry[]>([]);
  const [heapBytes, setHeapBytes] = useState<number | null>(null);
  const [namespaceOpen, setNamespaceOpen] = useState(false);

  // Caret offset tracked for the cell label only; actual cell is read from the
  // editor ref at run time so a stale render can never send the wrong cell.
  const editorRef = useRef<CodeEditorHandle>(null);
  const [cursorOffset, setCursorOffset] = useState(0);
  const cells = useMemo(() => splitPythonCells(code), [code]);
  const activeCellIndex = useMemo(
    () => cellIndexAtOffset(cells, cursorOffset),
    [cells, cursorOffset],
  );
  const ownsOutput = outputBufferId === buffer.id;

  /* Follow the caret — but only on genuine cell movement (not on cellRuns
   * updates, which would undo the run's own display choice) and only into cells
   * that have output (entering an unrun cell must not blank the pane). */
  const lastCaretCellRef = useRef(activeCellIndex);
  useEffect(() => {
    if (lastCaretCellRef.current === activeCellIndex) return;
    lastCaretCellRef.current = activeCellIndex;
    if (ownsOutput && cellRuns.has(activeCellIndex)) setDisplayedCell(activeCellIndex);
  }, [activeCellIndex, ownsOutput, cellRuns, setDisplayedCell]);

  const shownRun = !ownsOutput || displayedCell === null
    ? null
    : cellRuns.get(displayedCell) ?? null;
  const runResult = shownRun?.result ?? null;
  const resultFrame = shownRun?.frame ?? null;
  /* Output is stale if the cell was edited or a cell was inserted above it
   * (shifting indices); either means the on-screen result no longer matches the code. */
  const shownRunStale =
    shownRun !== null &&
    displayedCell !== null &&
    (cells[displayedCell]?.code ?? null) !== shownRun.code;

  /* Full console from the last run, in cell order. Run-scoped, not cell-scoped:
   * printing in one cell and drawing in the next is common. */
  const consoleOutput = useMemo(() => {
    if (!ownsOutput) return { stdout: "", stderr: "" };
    const parts = lastRunCellIndices
      .map((index) => cellRuns.get(index))
      .filter((entry): entry is CellRun => entry !== undefined);
    return {
      /* Budget refusal prepended to stdout: the live stream is reset per cell
       * so a pre-cell notice would be wiped, and a NameError a few lines below
       * needs this text to explain why the field was not bound. */
      stdout: skippedFieldsNotice(skippedFields) + parts.map((entry) => entry.result.stdout).join(""),
      stderr: parts.map((entry) => entry.result.stderr).join(""),
    };
  }, [ownsOutput, lastRunCellIndices, cellRuns, skippedFields]);

  /* Live output: chunks land in a ref; a timer publishes them to bound renders.
   * The timer reads the ref rather than closing over a value so late chunks
   * are included even if the run has already finished. */
  const streamRef = useRef<LiveStream | null>(null);
  const streamFlushRef = useRef<number | null>(null);
  const streamDatasetTokenRef = useRef<string | null>(null);
  const fieldReadControllersRef = useRef(new Set<AbortController>());

  // Plugin gallery: .py files in ~/.nimbus/plugins/, lazy-loaded on first open.
  const [galleryOpen, setGalleryOpen] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    try {
      return window.localStorage.getItem(GALLERY_OPEN_STORAGE_KEY) === "1";
    } catch {
      return false;
    }
  });
  const [pluginsDir, setPluginsDir] = useState<string | null>(null);
  const [plugins, setPlugins] = useState<PluginEntry[]>([]);
  const [pluginsLoading, setPluginsLoading] = useState(false);
  const [pluginsError, setPluginsError] = useState<string | null>(null);
  const [activePluginPath, setActivePluginPath] = useState<string | null>(null);

  const persistTimerRef = useRef<number | null>(null);
  // Ref mirror so the empty-deps unmount-flush can read the latest workspace.
  const workspaceRef = useRef(workspace);
  workspaceRef.current = workspace;

  // Debounced localStorage write while typing.
  useEffect(() => {
    if (persistTimerRef.current != null) {
      window.clearTimeout(persistTimerRef.current);
    }
    persistTimerRef.current = window.setTimeout(() => {
      setPersistError(persistWorkspace(loadedKeyRef.current, workspace));
    }, PERSIST_DEBOUNCE_MS) as unknown as number;
    return () => {
      if (persistTimerRef.current != null) {
        window.clearTimeout(persistTimerRef.current);
      }
    };
  }, [workspace]);

  // Unmount flush: empty deps so cleanup only runs at unmount, not on edits.
  useEffect(() => {
    return () => {
      persistWorkspace(loadedKeyRef.current, workspaceRef.current);
    };
  }, []);

  /* Swap workspaces on project change. The pending debounce must be cancelled
   * explicitly: the write effect's cleanup only fires on workspace changes, so
   * the in-flight timer would otherwise fire after loadedKeyRef moved on and
   * write the outgoing project's buffers into the incoming project's store. */
  useEffect(() => {
    if (loadedKeyRef.current === storageKey) return;
    if (persistTimerRef.current != null) {
      window.clearTimeout(persistTimerRef.current);
      persistTimerRef.current = null;
    }
    persistWorkspace(loadedKeyRef.current, workspaceRef.current);
    loadedKeyRef.current = storageKey;

    const loaded = loadPersistedWorkspace(storageKey, seedKey);
    setWorkspace(loaded.workspace);
    if (streamFlushRef.current != null) window.clearTimeout(streamFlushRef.current);
    streamFlushRef.current = null;
    streamRef.current = null;
    streamDatasetTokenRef.current = null;
    setPyStatus({ kind: "idle" });
    setNamespace([]);
    setHeapBytes(null);
    setLoadedPackages(new Set());
    setMissingModule(null);
    setPackageError(null);
    setPackageLoading(false);
    setInstallingPackage(null);
    setCanTryMicropip(false);
    setStagingBusy(false);
    setStagingError(null);
    setPersistError(null);
    setIoStatus(loaded.seeded
      ? "Opened the scripts you had open, as this project's own workspace. Unsaved scratch stays where it was written."
      : null);
  }, [storageKey, seedKey, setIoStatus]);

  /** Open a path as a tab, reusing an existing tab for the same file.
   *  Named differently from the Tauri bridge's openPath (OS file-manager reveal). */
  const openScriptPath = useCallback(async (path: string) => {
    const owner = workspaceOperationRef.current.capture();
    if (owner.key !== loadedKeyRef.current) return;
    setIoStatus(null);
    try {
      const text = await readPySnippet(path);
      if (!workspaceOperationRef.current.current(owner)) return;
      let reused = false;
      setWorkspace((current) => {
        const result = openPathInWorkspace(current, path, text);
        reused = !result.opened;
        return result.workspace;
      });
      const base = fileNameOf(path);
      setIoStatus(reused ? `Switched to ${base}` : `Opened ${base}`);
    } catch (e) {
      if (!workspaceOperationRef.current.current(owner)) return;
      setIoStatus(`Open failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }, [setIoStatus]);

  const onLoadSnippet = useCallback(async () => {
    const owner = workspaceOperationRef.current.capture();
    setIoStatus(null);
    try {
      const path = await pickPySnippetLoadPath();
      if (!path || !workspaceOperationRef.current.current(owner)) return;
      await openScriptPath(path);
    } catch (e) {
      if (!workspaceOperationRef.current.current(owner)) return;
      setIoStatus(`Open failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }, [openScriptPath, setIoStatus]);

  const refreshPlugins = useCallback(async () => {
    setPluginsLoading(true);
    setPluginsError(null);
    try {
      const [dir, list] = await Promise.all([getPluginsDir(), listPlugins()]);
      setPluginsDir(dir);
      setPlugins(list);
    } catch (e) {
      setPluginsError(e instanceof Error ? e.message : String(e));
    } finally {
      setPluginsLoading(false);
    }
  }, []);

  // Lazy-load the gallery on first open; refresh after save so new files appear.
  useEffect(() => {
    if (galleryOpen) {
      void refreshPlugins();
    }
    try {
      window.localStorage.setItem(
        GALLERY_OPEN_STORAGE_KEY,
        galleryOpen ? "1" : "0",
      );
    } catch {
      // Persistence best-effort.
    }
  }, [galleryOpen, refreshPlugins]);

  /** Save As — always prompts, and adopts the chosen path for later saves. */
  const onSaveSnippetAs = useCallback(async () => {
    const owner = workspaceOperationRef.current.capture();
    if (owner.key !== loadedKeyRef.current) return;
    setIoStatus(null);
    const target = activeBuffer(workspaceRef.current);
    try {
      // Default save target is the plugins dir so newly-saved snippets
      // land in the gallery on the next refresh. Falls back to the
      // bare filename when getPluginsDir hasn't resolved yet (e.g. in
      // browser/mock mode).
      let defaultName = target.path ?? "nimbus_snippet.py";
      if (!target.path) {
        try {
          const dir = pluginsDir ?? (await getPluginsDir());
          if (!pluginsDir) setPluginsDir(dir);
          defaultName = `${dir}/${defaultName}`;
        } catch {
          // Plugins dir unavailable; fall through with bare filename.
        }
      }
      const path = await pickPySnippetSavePath(defaultName);
      if (!path || !workspaceOperationRef.current.current(owner)) return;
      await writePySnippet(path, target.code);
      if (!workspaceOperationRef.current.current(owner)) return;
      setWorkspace((current) => markBufferSaved(current, target.id, path, target.code));
      setIoStatus(`Saved ${fileNameOf(path)}`);
      setActivePluginPath(path);
      // If the save landed in the plugins dir, the gallery list is stale.
      // Refresh so the user sees their new entry immediately.
      if (galleryOpen) {
        void refreshPlugins();
      }
    } catch (e) {
      if (!workspaceOperationRef.current.current(owner)) return;
      setIoStatus(`Save failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }, [pluginsDir, galleryOpen, refreshPlugins, setIoStatus]);

  /* Save: write in-place when the path is known, otherwise delegate to Save As. */
  const onSaveSnippet = useCallback(async () => {
    const owner = workspaceOperationRef.current.capture();
    if (owner.key !== loadedKeyRef.current) return;
    const target = activeBuffer(workspaceRef.current);
    if (!saveWritesInPlace(target)) {
      await onSaveSnippetAs();
      return;
    }
    setIoStatus(null);
    try {
      await writePySnippet(target.path, target.code);
      if (!workspaceOperationRef.current.current(owner)) return;
      setWorkspace((current) => markBufferSaved(current, target.id, target.path, target.code));
      setIoStatus(`Saved ${target.name}`);
      if (galleryOpen) void refreshPlugins();
    } catch (e) {
      if (!workspaceOperationRef.current.current(owner)) return;
      setIoStatus(`Save failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }, [onSaveSnippetAs, galleryOpen, refreshPlugins, setIoStatus]);

  const onLoadPlugin = useCallback(async (plugin: PluginEntry) => {
    setActivePluginPath(plugin.path);
    await openScriptPath(plugin.path);
  }, [openScriptPath]);

  /** Focus a tab. Fetching the text, if it still needs fetching, is the
   *  reload effect's job — see below. */
  const onSelectBuffer = useCallback((id: string) => {
    setWorkspace((current) => activateBuffer(current, id));
  }, []);

  const onBufferTabKeyDown = useCallback((
    event: ReactKeyboardEvent<HTMLButtonElement>,
    index: number,
  ) => {
    const entries = workspaceRef.current.buffers;
    let nextIndex: number | null = null;
    if (event.key === "ArrowRight") nextIndex = (index + 1) % entries.length;
    else if (event.key === "ArrowLeft") nextIndex = (index - 1 + entries.length) % entries.length;
    else if (event.key === "Home") nextIndex = 0;
    else if (event.key === "End") nextIndex = entries.length - 1;
    if (nextIndex === null) return;
    event.preventDefault();
    const next = entries[nextIndex];
    setWorkspace((current) => activateBuffer(current, next.id));
    window.requestAnimationFrame(() => bufferTabRefs.current.get(next.id)?.focus());
  }, []);

  /* Fetch text for a path-only buffer (recovery store does not duplicate files).
   * needsReload stays set until text lands to prevent saving an empty buffer
   * back over the real file. */
  const { id: focusedId, name: focusedName, path: focusedPath, needsReload } = buffer;
  useEffect(() => {
    if (!needsReload || !focusedPath) return;
    const [id, path, name] = [focusedId, focusedPath, focusedName];
    let cancelled = false;
    void (async () => {
      setReloadFailedIds((current) => {
        if (!current.has(id)) return current;
        const next = new Set(current);
        next.delete(id);
        return next;
      });
      try {
        const text = await readPySnippet(path);
        if (cancelled) return;
        setWorkspace((current) => {
          // Typing clears needsReload; skip the disk read if that happened.
          const target = current.buffers.find((entry) => entry.id === id);
          return target?.needsReload ? markBufferLoaded(current, id, text) : current;
        });
        setReloadFailedIds((current) => {
          if (!current.has(id)) return current;
          const next = new Set(current);
          next.delete(id);
          return next;
        });
      } catch (e) {
        if (!cancelled) {
          setReloadFailedIds((current) => new Set(current).add(id));
          setIoStatus(`Could not reopen ${name}: ${e instanceof Error ? e.message : String(e)}`);
        }
      }
    })();
    return () => { cancelled = true; };
  }, [focusedId, focusedName, focusedPath, needsReload, setIoStatus]);

  const onCloseBuffer = useCallback((id: string) => {
    const target = workspaceRef.current.buffers.find((entry) => entry.id === id);
    // Unsaved scratch has no disk copy; confirm before discarding it.
    if (target && isBufferDirty(target)) {
      const label = target.path ? target.name : `“${target.name}”`;
      if (!window.confirm(`${label} has unsaved changes. Close and lose them?`)) return;
    }
    setReloadFailedIds((current) => {
      if (!current.has(id)) return current;
      const next = new Set(current);
      next.delete(id);
      return next;
    });
    focusBufferAfterCloseRef.current = true;
    setWorkspace((current) => closeBuffer(current, id, STARTER_CODE));
  }, []);

  useEffect(() => {
    if (!focusBufferAfterCloseRef.current) return;
    focusBufferAfterCloseRef.current = false;
    bufferTabRefs.current.get(workspace.activeId)?.focus();
  }, [workspace.activeId, workspace.buffers]);

  const onAddBuffer = useCallback(() => {
    setWorkspace((current) => {
      if (!canAddBuffer(current)) {
        setIoStatus(`At most ${MAX_BUFFERS} tabs can be open.`);
        return current;
      }
      return addScratchBuffer(current, STARTER_CODE);
    });
  }, [setIoStatus]);

  const onUseStarter = useCallback(() => {
    const target = activeBuffer(workspaceRef.current);
    if (
      isBufferDirty(target)
      && !window.confirm(`${target.name} has unsaved changes. Replace them with the starter script?`)
    ) return;
    setCode(STARTER_CODE);
    editorRef.current?.focus();
  }, [setCode]);

  const onOpenPluginsDir = useCallback(async () => {
    try {
      const dir = pluginsDir ?? (await getPluginsDir());
      if (!pluginsDir) setPluginsDir(dir);
      await openPath(dir);
    } catch (e) {
      setPluginsError(e instanceof Error ? e.message : String(e));
    }
  }, [pluginsDir]);

  const datasetToken = JSON.stringify(sourceHandles.map((source) => [
    source.role,
    source.id,
    source.runDir,
    source.steps[0]?.publicationGeneration ?? null,
  ]));
  useEffect(() => () => {
    for (const controller of fieldReadControllersRef.current) controller.abort();
    fieldReadControllersRef.current.clear();
  }, [datasetToken, runKey]);
  /* A new dataset or republished run invalidates all stored outputs. Guard
   * against a token match on mount so a run that finished while navigating
   * away is not thrown away the moment the user returns. */
  useEffect(() => {
    if (readSession(runKey).datasetToken === datasetToken) return;
    patchSession(runKey, {
      datasetToken,
      running: false,
      stopStage: "idle",
      outputBufferId: null,
      cellRuns: new Map(),
      displayedCell: null,
      lastRunCellIndices: [],
      liveStream: null,
      skippedFields: NO_SKIPPED_FIELDS,
    });
  }, [datasetToken, runKey]);
  const stepFieldNames = useMemo<string[]>(
    () => currentStep?.fields ?? [],
    [currentStep],
  );

  /** Field names referenced in executable code (comments and strings excluded),
   *  so documentation prose does not trigger field loads. */
  const referencedFields = useMemo(
    () => detectReferencedFields(code, stepFieldNames),
    [code, stepFieldNames],
  );

  const pushStream = useCallback((stream: "stdout" | "stderr", text: string) => {
    const current = streamRef.current ?? { stdout: "", stderr: "", truncated: false };
    const appended = appendCapped(current[stream], text);
    streamRef.current = {
      ...current,
      [stream]: appended.text,
      truncated: current.truncated || appended.truncated,
    };
    if (streamFlushRef.current !== null) return;
    streamFlushRef.current = window.setTimeout(() => {
      streamFlushRef.current = null;
      // Read through the ref: chunks that arrived while this was pending must
      // be included, and the run may have finished and cleared it entirely.
      const token = streamDatasetTokenRef.current;
      if (token !== null && readSession(runKey).datasetToken === token) {
        patchSession(runKey, {
          liveStream: streamRef.current ? { ...streamRef.current } : null,
        });
      }
    }, STREAM_FLUSH_MS);
  }, [runKey]);

  const cancelStreamFlush = useCallback(() => {
    if (streamFlushRef.current !== null) {
      window.clearTimeout(streamFlushRef.current);
      streamFlushRef.current = null;
    }
  }, []);

  // A pending flush after unmount would setState on a dead component.
  useEffect(() => cancelStreamFlush, [cancelStreamFlush]);

  /* Run a sequence of cells, one interpreter call each. Boot, field load, and
   * bind happen once per run; each cell gets its own result entry. */
  const runCells = useCallback(async (targets: readonly PythonCell[]) => {
    /* Multi-cell sequences skip empty cells (comment-only blocks); single-cell
     * runs always execute so the user gets feedback even on an empty cell.
     * A missing frame is not a reason to refuse: fields are simply absent and
     * Python still runs, which lets the starter figure work with no dataset open. */
    const sequence = targets.length > 1 ? targets.filter((cell) => !isCellEmpty(cell)) : targets;
    if (sequence.length === 0) return;
    // Claim before any field read or context bind. Rapid shortcuts and toolbar
    // clicks must never interleave two transactions inside the same pane.
    if (!runtime.tryAcquireExecution()) return;
    const executionDatasetToken = datasetToken;
    const executionUiOwner = workspaceOperationRef.current.capture();
    const currentUi = () => workspaceOperationRef.current.current(executionUiOwner);
    const updatePyStatus = (status: PyodideStatus) => { if (currentUi()) setPyStatus(status); };
    const executionBufferId = workspaceRef.current.activeId;
    const commitSession = (
      update: Partial<PythonRunSession>
        | ((previous: PythonRunSession) => Partial<PythonRunSession>),
    ) => patchSession(runKey, (previous) => {
      if (
        previous.datasetToken !== executionDatasetToken
        || previous.outputBufferId !== executionBufferId
      ) return {};
      return typeof update === "function" ? update(previous) : update;
    });
    const isCurrentRun = () => {
      const current = readSession(runKey);
      return current.datasetToken === executionDatasetToken
        && current.outputBufferId === executionBufferId;
    };
    patchSession(runKey, (previous) => ({
      ...(previous.datasetToken === executionDatasetToken
          && previous.outputBufferId === executionBufferId
        ? {}
        : {
            datasetToken: executionDatasetToken,
            cellRuns: new Map(),
            displayedCell: null,
            lastRunCellIndices: [],
          }),
      outputBufferId: executionBufferId,
      running: true,
      loadErr: null,
    }));
    setPackageError(null);
    streamRef.current = { stdout: "", stderr: "", truncated: false };
    streamDatasetTokenRef.current = executionDatasetToken;
    commitSession({ liveStream: streamRef.current });
    const autoLoadController = new AbortController();
    fieldReadControllersRef.current.add(autoLoadController);
    try {
      // Boot/refresh Pyodide. The first run can take 5-10 s while the
      // WASM blob downloads; subsequent runs are sub-second.
      const handle = await runtime.getPyodide(updatePyStatus);
      if (!isCurrentRun()) return;

      /* Load each referenced field. Empty when no step is open, so the loop
       * is a no-op and Python raises NameError rather than refusing to start. */
      const loaded: { name: string; field: NpyN }[] = [];
      const skipped: SkippedField[] = [];
      let injectedBytes = 0;
      const fieldsForSequence = detectReferencedFields(
        sequence.map((cell) => cell.code).join("\n"),
        stepFieldNames,
      );
      for (const name of fieldsForSequence) {
        if (!currentStep) break;
        // Skip the read entirely once budget is gone: allocating to learn a size
        // that only appears in a "not bound" message is wasteful.
        if (budgetExhausted(injectedBytes)) {
          skipped.push({ name, bytes: null });
          continue;
        }
        try {
          const rank = pythonFieldRank(runDir, currentStep, name);
          const declaredShape = rank === 3
            ? [currentStep.grid?.nr, currentStep.grid?.nth, currentStep.grid?.nz]
            : rank === 2
              ? [currentStep.grid?.nr, currentStep.grid?.nth]
              : null;
          const qualifiedDeclaredShape = declaredShape?.every((count): count is number =>
            count != null && Number.isSafeInteger(count) && count > 0
          ) ? declaredShape : null;
          if (!qualifiedDeclaredShape) {
            throw new Error(
              `Cannot auto-load '${name}' because its rank/shape is not declared. `
                + `Use an explicit bounded load_field() request once the provider publishes metadata.`,
            );
          }
          const estimatedBytes = qualifiedDeclaredShape
            .reduce((product, count) => product * count, 1)
            * Float32Array.BYTES_PER_ELEMENT;
          if (!fitsInBudget(injectedBytes, estimatedBytes)) {
            skipped.push({ name, bytes: estimatedBytes });
            continue;
          }
          const field = await readFieldAny(
            runDir,
            currentStep.index,
            name,
            currentStep.publicationGeneration,
            { signal: autoLoadController.signal },
          );
          if (!isCurrentRun()) return;
          /* Budget enforced on actual read size, not a shape estimate (column
           * diagnostics are 2-D; a 3-D guess would wrongly refuse them).
           * Reading then declining still wins: the read populates the frame
           * cache while binding copies into the 4 GiB wasm heap. */
          const bytes = field.data.byteLength;
          if (!fitsInBudget(injectedBytes, bytes)) {
            skipped.push({ name, bytes });
            continue;
          }
          injectedBytes += bytes;
          loaded.push({ name, field });
        } catch (e) {
          throw new Error(
            `Couldn't load field '${name}': ${e instanceof Error ? e.message : String(e)}`,
          );
        }
      }
      // Recorded so the console can prepend it; a NameError below needs this context.
      commitSession({ skippedFields: skipped.length > 0 ? skipped : NO_SKIPPED_FIELDS });

      await runtime.bindContext(handle, {
        fields: loaded,
        grid: currentStep?.grid ?? null,
        simulationTimeS: currentStep?.simulationTimeS ?? null,
        fieldUnits: currentStep?.fieldUnits,
        frameIndex: currentStep ? stepIdx : undefined,
        stepIndex: currentStep?.index,
        frameCount: steps.length,
        frameTimesS: steps.map((step) => step.simulationTimeS),
        sources: sourceHandles.map((handle) => ({
          role: handle.role,
          displayName: handle.displayName,
          frameCount: handle.steps.length,
        })),
        loadFrameField: async (fieldName, frameIndex, source, window) => {
          const controller = new AbortController();
          fieldReadControllersRef.current.add(controller);
          try {
          const handle = sourceHandles.find((entry) => entry.role === source);
          if (!handle) throw new Error(`No open dataset for source '${source}'`);
          const target = handle.steps[frameIndex];
          if (!target) throw new Error(`No frame ${frameIndex} in source '${source}'`);
          const rank = pythonFieldRank(handle.runDir, target, fieldName);
          const sourceShape = rank === 3
            ? [target.grid?.nr, target.grid?.nth, target.grid?.nz]
            : rank === 2
              ? [target.grid?.nr, target.grid?.nth]
              : null;
          const qualifiedShape = sourceShape?.every((count): count is number =>
            count != null && Number.isSafeInteger(count) && count > 0
          ) ? sourceShape : null;
          if (qualifiedShape && window) {
            if (window.stride.length < qualifiedShape.length
                || window.stride.some((step) => !Number.isSafeInteger(step) || step <= 0)) {
              throw new Error(
                `stride for rank-${qualifiedShape.length} '${fieldName}' must contain `
                  + `${qualifiedShape.length} positive integers`,
              );
            }
            if (window.box != null && (
              window.box.length !== qualifiedShape.length
              || window.box.some(([start, stop]) =>
                !Number.isSafeInteger(start) || !Number.isSafeInteger(stop) || stop <= start)
            )) {
              throw new Error(
                `box for rank-${qualifiedShape.length} '${fieldName}' must contain `
                  + `${qualifiedShape.length} increasing (start, stop) pairs`,
              );
            }
          }
          const effectiveWindow = qualifiedShape
            ? pythonReadWindow(qualifiedShape, window, target)
            : window;
          if (qualifiedShape && fieldSupportsBoundedWindows(target.fieldOrigins?.[fieldName])) {
            const terrainRemap = effectiveWindow?.terrainRemap === true
              && target.grid?.terrainFollowing === true
              && qualifiedShape.length === 3;
            // Native terrain windows include geometry even when Python keeps model levels.
            const outputBytesPerCell = target.grid?.terrainFollowing && qualifiedShape.length === 3
              ? PYTHON_TERRAIN_BYTES_PER_CELL
              : Float32Array.BYTES_PER_ELEMENT;
            const requestsPointSamples = !terrainRemap && effectiveWindow?.stride.some(
              (step) => Math.trunc(step) > 1,
            ) === true;
            let plan = planPythonFieldTiles(
              qualifiedShape,
              effectiveWindow,
              undefined,
              outputBytesPerCell,
              terrainRemap ? fieldWindowAggregation(fieldName) : "point_sample",
            );
            if (plan.outputBytes > MAX_AUTO_INJECT_BYTES) {
              throw new Error(
                `Requested Python field window needs ${formatBytes(plan.outputBytes)}; `
                  + `the per-field limit is ${formatBytes(MAX_AUTO_INJECT_BYTES)}.`,
              );
            }
            let assembly = createSubsampledWindowAssembly(qualifiedShape, effectiveWindow);
            const readPlan = async (): Promise<void> => {
              for (const tile of plan.windows) {
                const chunk = await readFieldAnyWindows(
                  handle.runDir,
                  target.index,
                  fieldName,
                  target.publicationGeneration,
                  [tile],
                  { signal: controller.signal },
                );
                assembly.addTile(chunk, tile);
              }
            };
            try {
              await readPlan();
            } catch (error) {
              if (!requestsPointSamples || !needsDensePythonFallback(error)) throw error;
              // A sparse request can span more compressed bricks or exact
              // geometry than one native budget. Split it into bounded tiles
              // while retaining the same point-sample contract.
              plan = planPythonDenseFieldTiles(
                qualifiedShape,
                effectiveWindow,
                undefined,
                outputBytesPerCell,
              );
              assembly = createSubsampledWindowAssembly(qualifiedShape, effectiveWindow);
              await readPlan();
            }
            const prepared = remapPythonTerrainField(
              assembly.field,
              target,
              effectiveWindow?.terrainRemap,
            );
            return {
              ...prepared.field,
              pythonMetadata: pythonFieldMetadata(
                prepared.field,
                prepared.terrainRemapped ? prepared.field.shape : qualifiedShape,
                prepared.terrainRemapped
                  ? { stride: [], box: null, validOnly: effectiveWindow?.validOnly }
                  : effectiveWindow ?? { stride: [], box: null },
                prepared.metadataStep,
                frameIndex,
                fieldName,
                handle,
                prepared.terrainRemapped,
                {
                  sourceShape: qualifiedShape,
                  window: effectiveWindow ?? { stride: [], box: null },
                  geometry: assembly.field.windowGeometry,
                },
              ),
            };
          }
          if (!qualifiedShape) {
            throw new Error(`Cannot determine a bounded rank-2/3 shape for '${fieldName}'.`);
          }
          const requestsSubset = effectiveWindow?.box != null
            || effectiveWindow?.stride.some((step) => Math.trunc(step) > 1) === true;
          if (requestsSubset) {
            throw new Error(
              `'${fieldName}' is supplied by a provider that cannot apply Python windows at the reader.`,
            );
          }
          if (effectiveWindow?.terrainRemap && target.grid?.terrainFollowing) {
            throw new Error(
              `'${fieldName}' cannot be terrain-remapped because its provider does not serve bounded physical geometry.`,
            );
          }
          const sourceBytes = qualifiedShape.reduce((product, count) => product * count, 1)
            * Float32Array.BYTES_PER_ELEMENT;
          if (sourceBytes > MAX_AUTO_INJECT_BYTES) {
            throw new Error(
              `Whole-field Python load needs ${formatBytes(sourceBytes)}; `
                + `the per-field limit is ${formatBytes(MAX_AUTO_INJECT_BYTES)}.`,
            );
          }
          const field = await readFieldAny(
            handle.runDir,
            target.index,
            fieldName,
            target.publicationGeneration,
            { signal: controller.signal },
          );
          return {
            ...field,
            pythonMetadata: pythonFieldMetadata(
              field,
              qualifiedShape,
              effectiveWindow ?? { stride: [], box: null },
              target,
              frameIndex,
              fieldName,
              handle,
            ),
          };
          } finally {
            fieldReadControllersRef.current.delete(controller);
          }
        },
      });
      if (!isCurrentRun()) return;
      // Rebind every run so released files disappear from `files`.
      await bindStagedFiles(stagedFiles, {
        bind: runtime.bindStagedFilesInPython.bind(runtime),
      });

      // Null when no dataset is open: don't stamp a figure with a made-up frame.
      const frame: ResultFrame | null = currentStep
        ? {
          stepIndex: currentStep.index,
          simulationTimeS: currentStep.simulationTimeS,
          publicationGeneration: currentStep.publicationGeneration,
        }
        : null;

      let last: RunResult | null = null;
      let retainedRunBytes = 0;
      const ran: number[] = [];
      for (const [position, cell] of sequence.entries()) {
        // Each cell gets a fresh preview; without this the second cell's output
        // would appear appended to the first's while it ran.
        if (currentUi()) {
          streamRef.current = { stdout: "", stderr: "", truncated: false };
          commitSession({ liveStream: streamRef.current });
        }

        const rawResult = await runtime.runPythonCell(handle, cellSourceForExecution(cell), {
          onStream: (stream, text) => { if (currentUi()) pushStream(stream, text); },
          // Only cell 0 of a full run clears produced files; per-cell clear
          // would delete a file written by the cell above.
          resetProducedFiles: position === 0 && cell.index === 0,
        });
        if (!isCurrentRun()) return;
        const retained = retainRunResult(
          rawResult,
          MAX_OUTPUT_TOTAL_BYTES - retainedRunBytes,
        );
        const result = retained.result;
        retainedRunBytes += retained.bytes;
        last = result;
        commitSession((current) => {
          const next = new Map(current.cellRuns);
          next.set(cell.index, { result, code: cell.code, frame });
          let remaining = MAX_OUTPUT_TOTAL_BYTES;
          const retentionOrder = [
            cell.index,
            ...Array.from(next.keys()).reverse().filter((index) => index !== cell.index),
          ];
          for (const index of retentionOrder) {
            const entry = next.get(index);
            if (!entry) continue;
            const bounded = retainRunResult(entry.result, remaining);
            remaining = Math.max(0, remaining - bounded.bytes);
            if (bounded.result !== entry.result) {
              next.set(index, { ...entry, result: bounded.result });
            }
          }
          return { cellRuns: next, displayedCell: cell.index };
        });
        ran.push(cell.index);
        // Stop at the first cell that RAISED. A cell that merely warned has
        // still produced its state, and the cells below it depend on that
        // state, so a warning must not halt the sequence.
        if (result.raised) break;
      }

      commitSession({ lastRunCellIndices: ran });
      if (last && currentUi()) {
        const hasConsole = Boolean(last.stdout.trim() || last.stderr.trim());
        setActivePane(last.stderr.trim() || (last.outputs.length === 0 && hasConsole)
          ? "output"
          : "result");
        setMissingModule(last.stderr ? detectMissingModule(last.stderr) : null);
      }
      // Refresh in parallel; import hooks may have loaded packages during the run.
      const [packages, names, heap] = await Promise.all([
        runtime.loadedPackageNames(),
        runtime.inspectNamespace(),
        runtime.interpreterHeapBytes(),
      ]);
      if (!isCurrentRun()) return;
      if (currentUi()) {
        setLoadedPackages(packages);
        setNamespace(names);
        setHeapBytes(heap);
      }
    } catch (e) {
      if (!isCurrentRun()) return;
      // PythonStoppedError: op rejected because Stop killed the thread — not a failure.
      if (e instanceof PythonStoppedError) {
        if (currentUi()) setMissingModule(null);
        return;
      }
      // PyodideFatalError: the interpreter is gone; needs its own banner, not a cell error.
      if (e instanceof PyodideFatalError) commitSession({ fatalError: e.message });
      else commitSession({ loadErr: e instanceof Error ? e.message : String(e) });
      if (currentUi()) setMissingModule(null);
    } finally {
      fieldReadControllersRef.current.delete(autoLoadController);
      // Cancel pending flush so a late timer doesn't re-show the partial tail.
      if (currentUi()) cancelStreamFlush();
      if (currentUi() && streamDatasetTokenRef.current === executionDatasetToken) {
        streamRef.current = null;
        streamDatasetTokenRef.current = null;
      }
      // Completion belongs to the run transaction, not to the output buffer.
      // Force-stop intentionally clears output ownership before the worker
      // rejects, and switching script tabs also hides the old buffer's output;
      // neither should suppress the run's one-and-only completion update.
      patchSession(runKey, (current) => current.datasetToken === executionDatasetToken
        ? {
            running: false,
            liveStream: null,
            completedRuns: current.completedRuns + 1,
          }
        : {});
      runtime.releaseExecution();
    }
  }, [
    runDir,
    runKey,
    currentStep,
    datasetToken,
    stepFieldNames,
    stepIdx,
    stagedFiles,
    steps,
    sourceHandles,
    runtime,
    pushStream,
    cancelStreamFlush,
    // Session writers listed explicitly: switching project or pane must rebuild
    // this closure so outputs go to the session they were started from.
  ]);

  const onRun = useCallback(async () => {
    await runCells(cells);
  }, [runCells, cells]);

  /** The cell the caret is in right now, read from the editor rather than state. */
  const caretCell = useCallback((): PythonCell => {
    const offset = editorRef.current?.getCursorOffset() ?? cursorOffset;
    return cells[cellIndexAtOffset(cells, offset)];
  }, [cells, cursorOffset]);

  const onRunCell = useCallback(async () => {
    await runCells([caretCell()]);
  }, [runCells, caretCell]);

  const onRunCellAdvance = useCallback(async () => {
    const cell = caretCell();
    // The caret moves before the await, not after: a run can take seconds and
    // the user is already reading the next cell by the time it returns.
    const next = cells[cell.index + 1];
    if (next) editorRef.current?.setCursorOffset(next.startOffset);
    await runCells([cell]);
  }, [runCells, caretCell, cells]);

  /* Jump to the traceback line. Fragment padding makes those line numbers
   * refer to the actual editor file; without it the jump would land wrong. */
  const onJumpToLine = useCallback((line: number) => {
    editorRef.current?.setCursorOffset(offsetForLine(code, line));
  }, [code]);

  const hasClearableOutput = ownsOutput && (
    cellRuns.size > 0
    || lastRunCellIndices.length > 0
    || liveStream !== null
    || skippedFields.length > 0
    || loadErr !== null
  );
  const onClearOutput = useCallback(() => {
    if (running) return;
    patchSession(runKey, (previous) => {
      if (previous.outputBufferId !== buffer.id) return {};
      return {
        outputBufferId: null,
        cellRuns: new Map(),
        displayedCell: null,
        lastRunCellIndices: [],
        liveStream: null,
        skippedFields: NO_SKIPPED_FIELDS,
        loadErr: null,
      };
    });
    setMissingModule(null);
    setPackageError(null);
  }, [running, runKey, buffer.id]);

  const onOutputTabKeyDown = useCallback((event: ReactKeyboardEvent<HTMLButtonElement>) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const next = activePane === "result" ? "output" : "result";
    setActivePane(next);
    window.requestAnimationFrame(() => {
      document.getElementById(`${outputId}-${next}-tab`)?.focus();
    });
  }, [activePane, outputId]);

  const onRunAbove = useCallback(async () => {
    const cell = caretCell();
    if (cell.index === 0) return;
    // Every cell above, in order. The sequence stops at the first that raises:
    // the point of "run above" is to rebuild the state this cell depends on,
    // and carrying on past a failed dependency rebuilds it wrong.
    await runCells(cells.slice(0, cell.index));
  }, [runCells, caretCell, cells]);

  const onOpenDataFiles = useCallback(async () => {
    if (stagingBusy) return;
    const owner = workspaceOperationRef.current.capture();
    const current = () => workspaceOperationRef.current.current(owner);
    setStagingError(null);
    let paths: string[] = [];
    try {
      paths = await pickDataFiles();
    } catch (e) {
      if (current()) setStagingError(e instanceof Error ? e.message : String(e));
      return;
    }
    if (paths.length === 0 || !current()) return;

    setStagingBusy(true);
    try {
      // Boot early so data can be loaded before writing any code.
      await runtime.getPyodide(status => { if (current()) setPyStatus(status); });
      if (!current()) return;
      const added: StagedFile[] = [];
      let running = stagedBytes;
      const taken = new Set(stagedFiles.map((file) => file.name));
      for (const path of paths) {
        if (!current()) break;
        try {
          const file = await stageFile(path, taken, running, {
            write: runtime.writeStagedFile.bind(runtime),
          });
          taken.add(file.name);
          running += file.byteLength;
          added.push(file);
        } catch (e) {
          // Report the first refusal; keep whatever succeeded already.
          if (current()) setStagingError(e instanceof Error ? e.message : String(e));
          break;
        }
      }
      if (added.length > 0) {
        setStagedFiles((current) => [...current, ...added]);
        if (current()) setPackagesOpen(true);
      }
    } catch (e) {
      if (current()) setStagingError(e instanceof Error ? e.message : String(e));
    } finally {
      if (current()) setStagingBusy(false);
    }
  }, [stagingBusy, stagedBytes, stagedFiles, runtime, setStagedFiles]);

  const onReleaseDataFile = useCallback(async (target: StagedFile) => {
    const owner = workspaceOperationRef.current.capture();
    const current = () => workspaceOperationRef.current.current(owner);
    setStagingError(null);
    try {
      await runtime.getPyodide(status => { if (current()) setPyStatus(status); });
      await unstageFile(target, {
        remove: runtime.removeStagedFile.bind(runtime),
      });
      setStagedFiles((current) => current.filter((file) => file.name !== target.name));
    } catch (e) {
      if (current()) setStagingError(e instanceof Error ? e.message : String(e));
    }
  }, [runtime, setStagedFiles]);

  /** Drop any armed escalation. Safe to call when none is armed. */
  const cancelStopEscalation = useCallback(() => {
    if (stopTimerRef.current === null) return;
    clearTimeout(stopTimerRef.current);
    stopTimerRef.current = null;
  }, []);

  /** Replace the interpreter and reset all state that described it.
   *  Called by Restart and the forceful half of Stop; shared to keep the
   *  state-clearing list in one place. */
  const discardInterpreter = useCallback(async (status: string) => {
    const owner = workspaceOperationRef.current.capture();
    const current = () => workspaceOperationRef.current.current(owner);
    cancelStopEscalation();
    for (const controller of fieldReadControllersRef.current) controller.abort();
    fieldReadControllersRef.current.clear();
    runtime.restartPython();
    /* Single patch: running and stopStage must move together — a moment where
     * running=false but stopStage!='idle' would trigger the polite-stop effect
     * with a misleading status message. */
    patchSession(runKey, {
      running: false,
      stopStage: "idle",
      fatalError: null,
      outputBufferId: null,
      cellRuns: new Map(),
      displayedCell: null,
      lastRunCellIndices: [],
      loadErr: null,
      liveStream: null,
      ioStatus: status,
    });
    setPyStatus({ kind: "idle" });
    setNamespace([]);
    setHeapBytes(null);
    setLoadedPackages(new Set());
    setMissingModule(null);

    if (stagedFiles.length > 0) {
      // Re-stage (not drop): a stale `files` dict that resolves to nothing
      // would fail silently inside user code, which is worse.
      setStagingBusy(true);
      try {
        const { kept, dropped } = await restageFiles(stagedFiles, {
          write: runtime.writeStagedFile.bind(runtime),
          bind: (files) => bindStagedFiles(files, {
            bind: runtime.bindStagedFilesInPython.bind(runtime),
          }),
        });
        setStagedFiles(kept);
        if (current()) setStagingError(dropped.length === 0
          ? null
          : `Could not reload ${dropped.map((entry) => entry.file.name).join(", ")}: `
            + dropped[0].reason);
      } finally {
        if (current()) setStagingBusy(false);
      }
    }
  }, [cancelStopEscalation, stagedFiles, runKey, runtime, setStagedFiles]);

  /** Stop: KeyboardInterrupt first (preserves namespace), then worker replace if
   *  the cell doesn't respond. Pressing again during the grace period forces
   *  immediately. */
  const onStop = useCallback(() => {
    if (!running) return;
    for (const controller of fieldReadControllersRef.current) controller.abort();
    fieldReadControllersRef.current.clear();
    const decision = decideStop({ stage: stopStage, canInterrupt: runtimeCanInterrupt });
    cancelStopEscalation();

    if (decision.action === "terminate") {
      void discardInterpreter(decision.status);
      return;
    }

    /* Only write the interrupt byte while a cell is in flight: writing it during
     * package load or namespace read would land inside Studio's own bookkeeping.
     * A false return means the byte wasn't shared; fall through to force-stop. */
    if (runtime.isCellInFlight() && !runtime.interruptPython()) {
      void discardInterpreter(
        decideStop({ stage: stopStage, canInterrupt: false }).status,
      );
      return;
    }
    setStopStage(decision.nextStage);
    setIoStatus(decision.status);
    if (decision.escalateAfterMs === null) return;
    stopTimerRef.current = setTimeout(() => {
      stopTimerRef.current = null;
      // The cell took the interrupt and the run has already unwound. Forcing
      // now would kill an interpreter that did as it was asked.
      if (!runningRef.current) return;
      void discardInterpreter(
        decideStop({ stage: "interrupting", canInterrupt: runtimeCanInterrupt }).status,
      );
    }, decision.escalateAfterMs);
  }, [
    running,
    stopStage,
    cancelStopEscalation,
    discardInterpreter,
    setIoStatus,
    setStopStage,
    runtime,
    runtimeCanInterrupt,
  ]);

  /* Cancel escalation when the run ends: a timer for a finished run must not
   * force-stop the next interpreter. The forceful path clears both running and
   * stopStage in one patch so this effect never sees a half-applied teardown. */
  useEffect(() => {
    if (running || stopStage === "idle") return;
    cancelStopEscalation();
    patchSession(runKey, { stopStage: "idle", ioStatus: STOP_STATUS_INTERRUPTED });
  }, [running, stopStage, cancelStopEscalation, runKey]);

  /* Nothing may fire into a component that is gone. */
  useEffect(() => cancelStopEscalation, [cancelStopEscalation]);

  /** Restart: replaces the interpreter to reclaim wasm heap memory (which only
   *  ever grows) and clear the namespace. */
  const onRestart = useCallback(async () => {
    if (
      !runtime.pythonNeedsRestart()
      && !window.confirm(
        (running
          ? "Restart the interpreter? The cell that is running will be cut off, and every "
          : "Restart the interpreter? Every ")
        + "variable, import and installed package is "
        + "discarded, and files the last run produced are lost unless you have already "
        + "saved them. Your scripts are not affected.",
      )
    ) return;

    await discardInterpreter("Interpreter restarted.");
  }, [running, discardInterpreter, runtime]);

  const onLoadMissing = useCallback(async () => {
    if (!missingModule || packageLoading) return;
    const owner = workspaceOperationRef.current.capture();
    const current = () => workspaceOperationRef.current.current(owner);
    setPackageLoading(true);
    setPackageError(null);
    setCanTryMicropip(false);
    try {
      await runtime.loadPyodidePackage(missingModule, status => { if (current()) setPyStatus(status); });
      if (!current()) return;
      setMissingModule(null);
      // Auto-retry: the user clicked Load because they wanted the cell
      // to work; running it again is the natural follow-up.
      await onRun();
    } catch (e) {
      if (!current()) return;
      setPackageError(e instanceof Error ? e.message : String(e));
      // Only offer the micropip path when loadPackage rejected because
      // the package isn't in Pyodide's curated index. Other failures
      // (network, lock-file mismatch) wouldn't be helped by micropip.
      setCanTryMicropip(e instanceof MissingPackageError);
    } finally {
      if (current()) setPackageLoading(false);
    }
  }, [missingModule, packageLoading, onRun, runtime]);

  const onTryMicropip = useCallback(async () => {
    if (!missingModule || packageLoading) return;
    const owner = workspaceOperationRef.current.capture();
    const current = () => workspaceOperationRef.current.current(owner);
    setPackageLoading(true);
    setPackageError(null);
    setCanTryMicropip(false);
    try {
      await runtime.micropipInstall(missingModule, status => { if (current()) setPyStatus(status); });
      if (!current()) return;
      setMissingModule(null);
      await onRun();
    } catch (e) {
      if (!current()) return;
      setPackageError(e instanceof Error ? e.message : String(e));
    } finally {
      if (current()) setPackageLoading(false);
    }
  }, [missingModule, packageLoading, onRun, runtime]);

  const onDismissMissing = useCallback(() => {
    setMissingModule(null);
    setPackageError(null);
    setCanTryMicropip(false);
  }, []);

  /* Install a catalog package by choice, not from a traceback. Does not
   * re-run the cell — nothing failed, so there is nothing to retry. */
  const onInstallPackage = useCallback(async (importName: string) => {
    if (installingPackage) return false;
    const owner = workspaceOperationRef.current.capture();
    const current = () => workspaceOperationRef.current.current(owner);
    setInstallingPackage(importName);
    setPackageError(null);
    try {
      await runtime.installPackage(importName, status => { if (current()) setPyStatus(status); });
      if (!current()) return false;
      const packages = await runtime.loadedPackageNames();
      if (!current()) return false;
      setLoadedPackages(packages);
      // If this is what the last run tripped over, the banner is now stale.
      setMissingModule((current) => (current === importName ? null : current));
      return true;
    } catch (e) {
      if (!current()) return false;
      setPackageError(e instanceof Error ? e.message : String(e));
      return false;
    } finally {
      if (current()) setInstallingPackage(null);
    }
  }, [installingPackage, runtime]);

  const onInstallQueried = useCallback(async () => {
    const name = installQuery.trim();
    if (!name) return;
    if (await onInstallPackage(name)) setInstallQuery("");
  }, [installQuery, onInstallPackage]);

  /* Export an output to disk. Animations use the in-process GIF encoder
   * (no ffmpeg, no re-run; frames were rendered when the cell ran). */
  const onExportOutput = useCallback(async (
    output: PythonOutput,
    format?: ArrayExportFormat,
  ) => {
    setIoStatus(null);
    try {
      if (output.kind === "ndarray" && format) {
        const base = (output.label ?? "array").replace(/[^A-Za-z0-9._-]+/g, "_");
        const path = format === "npy"
          ? await pickExportSavePath(`${base}.npy`, "NumPy array", ["npy"])
          : await pickExportSavePath(`${base}.csv`, "CSV table", ["csv"]);
        if (!path) return;
        if (format !== "npy" && format !== "csv") throw new Error("Unsupported array export format");
        if (arrayExportControllerRef.current) throw new Error("Another array export is still encoding; wait or cancel it first");
        const controller = new AbortController();
        arrayExportControllerRef.current = controller;
        setArrayExporting(true);
        setIoStatus(`Encoding ${fileNameOf(path)}…`);
        try {
          const bytes = await pythonArrayExportService.encode(output.data, output.shape, format, output.dtype, controller.signal);
          if (controller.signal.aborted) throw new DOMException("Python array export cancelled", "AbortError");
          if (arrayExportControllerRef.current === controller) {
            arrayExportControllerRef.current = null;
            setArrayExporting(false);
          }
          await writeBinaryFile(path, bytes);
          setIoStatus(`Saved ${fileNameOf(path)} (${formatBytes(bytes.byteLength)})`);
        } finally {
          if (arrayExportControllerRef.current === controller) {
            arrayExportControllerRef.current = null;
            setArrayExporting(false);
          }
        }
        return;
      }
      if (output.kind === "image") {
        const svg = format === "svg" && output.svg != null;
        const path = await pickExportSavePath(svg ? "figure.svg" : "figure.png", svg ? "SVG figure" : "PNG image", [svg ? "svg" : "png"]);
        if (!path) return;
        await writeBinaryFile(path, svg ? new TextEncoder().encode(output.svg!) : base64ToBytes(output.pngBase64));
        setIoStatus(`Saved ${fileNameOf(path)}`);
        return;
      }
      if (output.kind === "animation") {
        const path = await pickExportSavePath("animation.gif", "GIF animation", ["gif"]);
        if (!path) return;
        setIoStatus(`Encoding ${output.framesBase64.length} frames…`);
        const bytes = await encodeFramesToGif(output.framesBase64, output.fps);
        await writeBinaryFile(path, bytes);
        setIoStatus(`Saved ${fileNameOf(path)} (${formatBytes(bytes.byteLength)})`);
      }
    } catch (e) {
      setIoStatus(e instanceof DOMException && e.name === "AbortError" ? "Array export cancelled." : `Export failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }, [setIoStatus]);

  /** Write a file the cell produced out to the real filesystem. */
  const onExportProduced = useCallback(async (file: ProducedFile) => {
    setIoStatus(null);
    try {
      const extension = file.name.includes(".") ? file.name.split(".").pop() as string : "";
      const path = await pickExportSavePath(
        file.name,
        extension ? `${extension.toUpperCase()} file` : "File",
        extension ? [extension] : [],
      );
      if (!path) return;
      const handle = await runtime.getPyodide(setPyStatus);
      await writeBinaryFile(path, await runtime.readProducedFile(handle, file));
      setIoStatus(`Saved ${fileNameOf(path)}`);
    } catch (e) {
      setIoStatus(`Export failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }, [setIoStatus, runtime]);

  // Cmd/Ctrl+Enter via onSubmit prop — fires only when the editor has focus.

  return (
    <div
      className={`python-plot ${galleryOpen ? "python-plot--with-gallery" : ""}`}
      /* Stable DOM attribute for browser specs to wait on — button labels
       * and pane contents change too, but this tracks running exactly. */
      data-running={running ? "1" : "0"}
      data-run-count={completedRuns}
      /* Custom property so the two CSS layouts (with/without gallery) own
       * their track lists and this only overrides the one number they differ on. */
      style={{ ["--python-editor-width" as string]: `${editorWidth}px` }}
    >
      {galleryOpen && (
        <PluginGallery
          plugins={plugins}
          dir={pluginsDir}
          loading={pluginsLoading}
          error={pluginsError}
          activePath={activePluginPath}
          onClose={() => setGalleryOpen(false)}
          onRefresh={refreshPlugins}
          onOpenDir={onOpenPluginsDir}
          onPick={onLoadPlugin}
        />
      )}
      <div className="python-plot__editor-pane">
        {/* Four rows: interpreter controls, file actions, cell bar, panel toggles —
            each row sits directly above the thing it acts on. */}
        <div className="python-plot__editor-header">
          <span className="python-plot__editor-label">Python (Pyodide)</span>
          <div className="python-plot__header-actions" role="toolbar" aria-label="Interpreter">
            <PyodideStatusBadge status={pyStatus} running={running} />
            <div className="python-plot__btn-group">
              <button
                type="button"
                className="python-plot__btn python-plot__btn--primary"
                onClick={onRun}
                disabled={running}
                title="Run the whole script (Ctrl/⌘+Shift+Enter)"
              >
                <Icon name="play" size={10} />
                <span>{running ? "Running…" : "Run script"}</span>
              </button>
              {/* Never disabled: terminating the worker always works regardless
                  of cross-origin isolation. */}
              <button
                type="button"
                className={`python-plot__btn ${
                  stopStage === "interrupting" ? "python-plot__btn--danger" : ""
                }`}
                onClick={onStop}
                disabled={!running}
                title={stopButtonTitle(stopStage, runtimeCanInterrupt)}
              >
                <Icon name="stop" size={10} />
                <span>{stopButtonLabel(stopStage)}</span>
              </button>
              {/* Enabled during a run: replacing the interpreter is the only way
                  out of a wedged cell. */}
              <button
                type="button"
                className={`python-plot__btn ${fatalError ? "python-plot__btn--primary" : ""}`}
                onClick={() => void onRestart()}
                title="Throw away the interpreter and start a fresh one. Variables, imports and installed packages are lost; your scripts are not."
              >
                <Icon name="refresh" size={10} />
                <span>Restart</span>
              </button>
            </div>
          </div>
        </div>
        <div className="python-plot__domain-row" aria-label="Python data source fidelity">
          <OutputHierarchyControls source={outputHierarchySource} />
          <PlotCompressionControls step={currentStep} fields={currentStep?.fields ?? []} />
        </div>
        {/* data-* attributes let tests assert the strip changed with project/pane. */}
        <div
          className="python-plot__tabs"
          data-script-scope={projectId ?? "none"}
          data-script-pane={paneId}
        >
          {/* Tabs scroll; file actions beside them do not. */}
          <div
            className="python-plot__tab-strip"
            role="tablist"
            aria-label={projectId ? "Open scripts for this project" : "Open scripts"}
          >
            {workspace.buffers.map((entry, index) => {
              const dirty = isBufferDirty(entry);
              return (
                <div
                  key={entry.id}
                  className={`python-plot__buffer-tab ${
                    entry.id === workspace.activeId ? "python-plot__buffer-tab--active" : ""
                  }`}
                >
                  <button
                    type="button"
                    role="tab"
                    aria-selected={entry.id === workspace.activeId}
                    aria-controls={editorId}
                    id={`${editorId}-${entry.id}`}
                    tabIndex={entry.id === workspace.activeId ? 0 : -1}
                    ref={(node) => {
                      if (node) bufferTabRefs.current.set(entry.id, node);
                      else bufferTabRefs.current.delete(entry.id);
                    }}
                    className="python-plot__buffer-name"
                    onClick={() => onSelectBuffer(entry.id)}
                    onKeyDown={(event) => onBufferTabKeyDown(event, index)}
                    title={entry.path ?? "Unsaved scratch buffer"}
                  >
                    {entry.name}
                    {dirty && (
                      <span className="python-plot__buffer-dirty" aria-label="unsaved">•</span>
                    )}
                  </button>
                  <button
                    type="button"
                    className="python-plot__buffer-close"
                    onClick={() => onCloseBuffer(entry.id)}
                    aria-label={`Close ${entry.name}`}
                    title={`Close ${entry.name}`}
                  >
                    <Icon name="x" size={9} />
                  </button>
                </div>
              );
            })}
            <button
              type="button"
              className="python-plot__buffer-add"
              onClick={onAddBuffer}
              disabled={!canAddBuffer(workspace)}
              aria-label="New scratch buffer"
              title={canAddBuffer(workspace) ? "New scratch buffer" : `Limit is ${MAX_BUFFERS} tabs`}
            >
              <Icon name="plus" size={10} />
            </button>
          </div>
          {/* File actions on the tab strip: each acts on the active tab. */}
          <div className="python-plot__tab-actions" role="toolbar" aria-label="Script file">
            <button
              type="button"
              className="python-plot__btn"
              onClick={onLoadSnippet}
              title="Open a .py file from disk"
            >
              <Icon name="archive" size={10} />
              <span>Open…</span>
            </button>
            <button
              type="button"
              className="python-plot__btn"
              onClick={onSaveSnippet}
              disabled={!isBufferDirty(buffer) && buffer.path !== null}
              title={saveWritesInPlace(buffer)
                ? `Save to ${buffer.path} (Ctrl/⌘+S)`
                : buffer.path
                  ? `${buffer.name} has not been read yet, so Save will ask where to write (Ctrl/⌘+S)`
                  : "Save to a new file (Ctrl/⌘+S)"}
            >
              <Icon name="image" size={10} />
              <span>{saveWritesInPlace(buffer) ? "Save" : "Save…"}</span>
            </button>
            <button
              type="button"
              className="python-plot__btn"
              onClick={onSaveSnippetAs}
              title="Save to a new file"
            >
              <span>Save as…</span>
            </button>
            <button
              type="button"
              className="python-plot__btn"
              onClick={onUseStarter}
              title="Reset to starter code"
            >
              <Icon name="plus" size={10} />
              <span>Starter</span>
            </button>
          </div>
        </div>
        <div className="python-plot__cell-bar" role="toolbar" aria-label="Cells">
          <span className="python-plot__cell-position">
            {cellLabel(cells[activeCellIndex], cells.length)}
            {cells.length > 1 && (
              <span className="python-plot__cell-count">
                {" "}
                · {activeCellIndex + 1} of {cells.length}
              </span>
            )}
          </span>
          {cells.length > 1 && (
            <>
              <button
                type="button"
                className="python-plot__btn"
                onClick={() => void onRunCell()}
                disabled={running || isCellEmpty(cells[activeCellIndex])}
                title={
                  isCellEmpty(cells[activeCellIndex])
                    ? "This cell has nothing to run"
                    : "Run the cell the caret is in (Ctrl/⌘+Enter)"
                }
              >
                <Icon name="play" size={10} />
                <span>Run cell</span>
              </button>
              <button
                type="button"
                className="python-plot__btn"
                onClick={() => void onRunCellAdvance()}
                disabled={running || isCellEmpty(cells[activeCellIndex])}
                title="Run this cell, then move to the next (⇧↵)"
              >
                <span>Run &amp; next</span>
              </button>
              <button
                type="button"
                className="python-plot__btn"
                onClick={() => void onRunAbove()}
                disabled={running || activeCellIndex === 0}
                title="Run every cell above this one, to rebuild what it depends on"
              >
                <span>Run above</span>
              </button>
              <button
                type="button"
                className="python-plot__btn"
                onClick={() => void onRun()}
                disabled={running}
                title="Run the whole script (Ctrl/⌘+Shift+Enter)"
              >
                <span>Run all</span>
              </button>
            </>
          )}
          <button
            type="button"
            className="python-plot__btn"
            onClick={() => editorRef.current?.insertAtCursor("\n# %% \n")}
            title="Split the script here. A # %% line starts a new cell and stays a plain comment, so the file still runs as an ordinary script."
          >
            <Icon name="plus" size={10} />
            <span>Cell break</span>
          </button>
          <span className="python-plot__cell-hint">
            {cells.length > 1
              ? "Ctrl/⌘+Enter cell · Shift+Enter next · Ctrl/⌘+Shift+Enter all"
              : "Ctrl/⌘+Enter run · split with # %%"}
          </span>
        </div>
        {/* Panel toggles: each expands the panel directly below it. */}
        <div className="python-plot__panel-bar" role="toolbar" aria-label="Panels">
          <button
            type="button"
            className={`python-plot__btn ${galleryOpen ? "python-plot__btn--active" : ""}`}
            onClick={() => setGalleryOpen((v) => !v)}
            title={galleryOpen ? "Hide plugin gallery" : "Show plugin gallery"}
            aria-expanded={galleryOpen}
          >
            <Icon name="archive" size={10} />
            <span>Gallery</span>
          </button>
          <button
            type="button"
            className={`python-plot__btn ${namespaceOpen ? "python-plot__btn--active" : ""}`}
            onClick={() => setNamespaceOpen((v) => !v)}
            title="Show what the interpreter currently holds"
            aria-expanded={namespaceOpen}
          >
            <Icon name="info" size={10} />
            <span>Variables</span>
          </button>
          <button
            type="button"
            className={`python-plot__btn ${packagesOpen ? "python-plot__btn--active" : ""}`}
            onClick={() => setPackagesOpen((v) => !v)}
            title={packagesOpen ? "Hide available packages" : "Show available packages"}
            aria-expanded={packagesOpen}
          >
            <Icon name="package" size={10} />
            <span>Packages</span>
          </button>
          <button
            type="button"
            className="python-plot__btn"
            onClick={onOpenDataFiles}
            disabled={stagingBusy}
            title="Copy a netCDF or npy file into the Python interpreter"
          >
            <Icon name="folder" size={10} />
            <span>{stagingBusy ? "Loading…" : "Open data"}</span>
          </button>
          {/* Warn when isolation is absent: Stop will replace the interpreter. */}
          {!runtimeCanInterrupt && (
            <span
              className="python-plot__panel-note"
              title={"SharedArrayBuffer is gated on COOP + COEP. Without those headers the "
                + "interrupt byte cannot cross into the interpreter's thread."}
            >
              <Icon name="warning" size={10} />
              Stop replaces the interpreter here — no cross-origin isolation
            </span>
          )}
        </div>
        {ioStatus && (
          <div
            className={`python-plot__io-status ${/^(?:Open|Save|Export) failed:|^Could not /.test(ioStatus)
              ? "python-plot__io-status--error"
              : ""}`}
            role={/^(?:Open|Save|Export) failed:|^Could not /.test(ioStatus) ? "alert" : "status"}
          >
            {ioStatus}
            {arrayExporting && <button type="button" className="python-plot__btn" onClick={() => arrayExportControllerRef.current?.abort()}>Cancel export</button>}
          </div>
        )}
        {recoveryWarning && (
          <div className="python-plot__package-error" role="alert">
            <span className="python-plot__package-error-text">{recoveryWarning}</span>
          </div>
        )}
        {persistError && (
          <div className="python-plot__package-error" role="alert">
            <span className="python-plot__package-error-text">{persistError}</span>
          </div>
        )}
        {stagingError && (
          <div className="python-plot__package-error" role="alert">
            <span className="python-plot__package-error-text">{stagingError}</span>
          </div>
        )}
        {namespaceOpen && (
          <div className="python-plot__packages">
            <p className="python-plot__packages-intro">
              <strong>Namespace</strong> — every cell shares this, so a name bound in
              one is visible in the next. Refreshed after each run.
              {heapBytes !== null && (
                <>
                  {" "}Interpreter memory: <strong>{formatBytes(heapBytes)}</strong> reserved.
                  WebAssembly memory only ever grows, so this is a high-water mark; if it
                  reaches what the browser will grant, the interpreter <em>aborts</em> rather
                  than raising. <code>del</code> hands an array back to Python to reuse, but
                  it cannot lower this number — only <strong>Restart</strong> can, by
                  replacing the thread that holds the memory.
                </>
              )}
            </p>
            {namespace.length === 0 ? (
              <p className="python-plot__packages-intro">
                Nothing bound yet — run a cell.
              </p>
            ) : (
              <ul className="python-plot__packages-list">
                {namespace.map((entry) => (
                  <li key={entry.name} className="python-plot__packages-item">
                    <code className="python-plot__packages-import">{entry.name}</code>
                    <span className="python-plot__packages-tag python-plot__packages-tag--bundled">
                      {entry.kind}
                    </span>
                    <span className="python-plot__packages-purpose">{entry.detail ?? ""}</span>
                    <span className="python-plot__packages-ready" />
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
        {packagesOpen && (
          <div className="python-plot__packages">
            {runResult && runResult.produced.length > 0 && (
              <>
                <p className="python-plot__packages-intro">
                  <strong>Produced by the last run</strong> — anything written to{" "}
                  <code>save_path(&quot;name&quot;)</code>, e.g.{" "}
                  <code>np.save(save_path(&quot;w.npy&quot;), w)</code> or{" "}
                  <code>ds.to_netcdf(save_path(&quot;out.nc&quot;))</code>.
                </p>
                <ul className="python-plot__packages-list">
                  {runResult.produced.map((file) => (
                    <li key={file.name} className="python-plot__staged-item">
                      <code className="python-plot__packages-import">{file.name}</code>
                      <span className="python-plot__packages-tag python-plot__packages-tag--preloaded">
                        {formatBytes(file.byteLength)}
                      </span>
                      <span className="python-plot__packages-purpose">produced by this run</span>
                      <button
                        type="button"
                        className="python-plot__btn"
                        onClick={() => void onExportProduced(file)}
                        disabled={file.byteLength > MAX_PRODUCED_FILE_BYTES}
                        title={file.byteLength > MAX_PRODUCED_FILE_BYTES
                          ? "This file exceeds the 256 MiB single-transfer export limit"
                          : `Save ${file.name}`}
                      >
                        Save…
                      </button>
                    </li>
                  ))}
                </ul>
              </>
            )}
            <p className="python-plot__packages-intro">
              <strong>Loaded files</strong> — reachable from Python as{" "}
              <code>files[&quot;name&quot;]</code>, e.g.{" "}
              <code>xr.open_dataset(files[&quot;cm1out.nc&quot;])</code> or{" "}
              <code>np.load(files[&quot;field.npy&quot;])</code>. Using{" "}
              {formatBytes(stagedBytes)} of {formatBytes(MAX_STAGED_TOTAL_BYTES)}; the
              interpreter holds a full copy, so release what you are done with.
            </p>
            {stagedFiles.length === 0 ? (
              <p className="python-plot__packages-intro">
                Nothing loaded yet — use <strong>Open data</strong> for a netCDF or npy file.
              </p>
            ) : (
              <ul className="python-plot__packages-list">
                {stagedFiles.map((file) => (
                  <li key={file.name} className="python-plot__staged-item">
                    <code className="python-plot__packages-import">{file.name}</code>
                    <span className="python-plot__packages-tag python-plot__packages-tag--bundled">
                      {formatBytes(file.byteLength)}
                    </span>
                    <span className="python-plot__packages-purpose" title={file.hostPath}>
                      {file.hostPath}
                    </span>
                    <button
                      type="button"
                      className="python-plot__btn"
                      onClick={() => void onReleaseDataFile(file)}
                      title={`Release ${file.name} from the interpreter`}
                    >
                      Release
                    </button>
                  </li>
                ))}
              </ul>
            )}
            <p className="python-plot__packages-intro">
              <code>numpy</code> and <code>matplotlib</code> are ready immediately.
              <strong> Bundled</strong> packages ship with the interpreter and load on
              demand. <strong>micropip</strong> entries are not in Pyodide's index and
              only install if PyPI has a pure-Python wheel. Installing here does not
              re-run your script.
            </p>
            <div className="python-plot__install-row">
              <input
                type="text"
                className="python-plot__install-input"
                value={installQuery}
                onChange={(event) => setInstallQuery(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    void onInstallQueried();
                  }
                }}
                placeholder="Install by name, e.g. metpy"
                aria-label="Install a package by name"
                disabled={installingPackage !== null}
              />
              <button
                type="button"
                className="python-plot__btn"
                onClick={() => void onInstallQueried()}
                disabled={installingPackage !== null || installQuery.trim().length === 0}
                title="Install anything not in the list below — bundled packages load directly, everything else goes through micropip"
              >
                {installingPackage === installQuery.trim() ? "Installing…" : "Install"}
              </button>
            </div>
            <ul className="python-plot__packages-list">
              {PYODIDE_CATALOG.map((entry) => {
                // Preloaded packages arrive with the interpreter, so they are
                // never actionable; everything else is "here" only once the
                // interpreter says so.
                const ready =
                  entry.availability === "preloaded" ||
                  loadedPackages.has(packageNameForImport(entry.importName));
                const busy = installingPackage === entry.importName;
                return (
                  <li key={entry.importName} className="python-plot__packages-item">
                    <code className="python-plot__packages-import">{entry.importName}</code>
                    <span
                      className={`python-plot__packages-tag python-plot__packages-tag--${
                        ready ? "loaded" : entry.availability
                      }`}
                    >
                      {ready ? "loaded" : entry.availability}
                    </span>
                    <span className="python-plot__packages-purpose">{entry.purpose}</span>
                    {ready ? (
                      // The "loaded" tag already says everything a button
                      // would, and a disabled button here reads as something
                      // the user failed to earn rather than something done.
                      <span className="python-plot__packages-ready" />
                    ) : (
                      <button
                        type="button"
                        className="python-plot__btn"
                        onClick={() => void onInstallPackage(entry.importName)}
                        disabled={installingPackage !== null}
                        title={
                          entry.availability === "bundled"
                            ? `Load ${packageNameForImport(entry.importName)} into the interpreter`
                            : `Install ${packageNameForImport(entry.importName)} from PyPI via micropip`
                        }
                      >
                        {busy ? "…" : entry.availability === "bundled" ? "Load" : "Install"}
                      </button>
                    )}
                  </li>
                );
              })}
            </ul>
          </div>
        )}
        {missingModule && (
          <div className="python-plot__missing-module">
            <Icon name="warning" size={12} />
            <span className="python-plot__missing-text">
              {isBundledPyodideImport(missingModule) ? (
                <>
                  Module <code>{missingModule}</code> isn't loaded yet. It ships with
                  Pyodide as <code>{packageNameForImport(missingModule)}</code>.
                </>
              ) : (
                <>
                  Module <code>{missingModule}</code> isn't in Pyodide's index. Loading
                  it can only work if PyPI has a pure-Python wheel.
                </>
              )}
            </span>
            <button
              type="button"
              className="python-plot__btn python-plot__btn--primary"
              onClick={onLoadMissing}
              disabled={packageLoading}
            >
              {packageLoading
                ? `Loading ${packageNameForImport(missingModule)}…`
                : `Load ${packageNameForImport(missingModule)}`}
            </button>
            <button
              type="button"
              className="python-plot__btn"
              onClick={onDismissMissing}
              disabled={packageLoading}
            >
              Dismiss
            </button>
          </div>
        )}
        {packageError && (
          <div className="python-plot__package-error" role="alert">
            <span className="python-plot__package-error-text">{packageError}</span>
            {canTryMicropip && missingModule && (
              <button
                type="button"
                className="python-plot__btn python-plot__btn--primary"
                onClick={onTryMicropip}
                disabled={packageLoading}
                title="Bootstrap micropip and install from PyPI"
              >
                {packageLoading
                  ? `Installing ${packageNameForImport(missingModule)}…`
                  : `Try micropip install ${packageNameForImport(missingModule)}`}
              </button>
            )}
          </div>
        )}

        {needsReload && !reloadFailedIds.has(focusedId) ? (
          <div
            id={editorId}
            className="python-plot__editor-loading"
            role="tabpanel"
            aria-labelledby={`${editorId}-${buffer.id}`}
            aria-busy="true"
            aria-live="polite"
          >
            Loading {buffer.name}…
          </div>
        ) : (
          <Suspense
            fallback={
              <div className="python-plot__editor-loading" role="status" aria-live="polite">
                Loading editor…
              </div>
            }
          >
            <CodeEditor
              ref={editorRef}
              id={editorId}
              ariaLabelledBy={`${editorId}-${buffer.id}`}
              ariaLabel={`Python editor for ${buffer.name}`}
              value={code}
              onChange={setCode}
              onSubmit={() => void onRunCell()}
              onSubmitAdvance={() => void onRunCellAdvance()}
              onRunAll={() => void onRun()}
              onSave={() => void onSaveSnippet()}
              onCursorChange={setCursorOffset}
              completeNames={completeNamesForRuntime}
              placeholder={needsReload
                ? `Could not reopen ${buffer.name}. Type to replace its contents.`
                : "# Type Python. Call display(x) to show a value."}
            />
          </Suspense>
        )}

        <div className="python-plot__editor-footer">
          {referencedFields.length > 0 ? (
            <span className="python-plot__refs">
              Auto-load: {referencedFields.join(", ")}
            </span>
          ) : (
            <span className="python-plot__refs python-plot__refs--muted">
              No field references detected yet.
            </span>
          )}
          {/* Fatal error styled apart: nothing else can succeed until restarted. */}
          {fatalError && (
            <div className="python-plot__fatal" role="alert">
              <Icon name="warning" size={12} />
              <span>{fatalError}</span>
              <button
                type="button"
                className="python-plot__btn python-plot__btn--primary"
                onClick={() => void onRestart()}
              >
                <Icon name="refresh" size={10} />
                <span>Restart interpreter</span>
              </button>
            </div>
          )}
          {loadErr && (
            <div className="python-plot__err" role="alert">
              <Icon name="warning" size={12} />
              <span>{loadErr}</span>
            </div>
          )}
        </div>
      </div>

      {/* Draggable editor/output split. */}
      <ResizeHandle
        onResize={adjustEditorWidth}
        placement="inline"
        label="Resize the code editor"
        minValuePx={EDITOR_WIDTH_MIN}
        maxValuePx={EDITOR_WIDTH_MAX}
      />

      <div className="python-plot__output-pane">
        <div className="python-plot__output-tabs">
          <div className="python-plot__output-tablist" role="tablist" aria-label="Python output">
            <button
              id={`${outputId}-result-tab`}
              type="button"
              role="tab"
              aria-selected={activePane === "result"}
              aria-controls={`${outputId}-result-panel`}
              tabIndex={activePane === "result" ? 0 : -1}
              className={`python-plot__tab ${activePane === "result" ? "python-plot__tab--active" : ""}`}
              onClick={() => setActivePane("result")}
              onKeyDown={onOutputTabKeyDown}
            >
              Result
            </button>
            <button
              id={`${outputId}-output-tab`}
              type="button"
              role="tab"
              aria-selected={activePane === "output"}
              aria-controls={`${outputId}-output-panel`}
              tabIndex={activePane === "output" ? 0 : -1}
              className={`python-plot__tab ${activePane === "output" ? "python-plot__tab--active" : ""}`}
              onClick={() => setActivePane("output")}
              onKeyDown={onOutputTabKeyDown}
            >
              stdout / stderr
              {/* Dot signals live activity or retained console text. */}
              {ownsOutput && (
                liveStream ? liveStream.stdout || liveStream.stderr : consoleOutput.stdout.trim() || consoleOutput.stderr.trim()
              ) && (
                <span
                  className="python-plot__tab-dot"
                  aria-label={liveStream ? "receiving output" : "has console output"}
                />
              )}
            </button>
          </div>
          {resultFrame && (
            <span className="python-plot__refs">
              {/* Cell provenance label so the user knows which cell's output is shown. */}
              {cells.length > 1 && displayedCell !== null && cells[displayedCell]
                ? `${cellLabel(cells[displayedCell], cells.length)} · `
                : ""}
              step {resultFrame.stepIndex}
              {resultFrame.simulationTimeS != null
                ? ` · t=${resultFrame.simulationTimeS.toFixed(1)} s`
                : ""}
              {currentStep && (
                currentStep.index !== resultFrame.stepIndex ||
                currentStep.publicationGeneration !== resultFrame.publicationGeneration
              ) ? " · selected frame differs" : ""}
            </span>
          )}
          {shownRunStale && (
            <span className="python-plot__stale" title="The cell has been edited since this ran">
              <Icon name="warning" size={10} />
              <span>code changed since this ran</span>
            </span>
          )}
          <button
            type="button"
            className="python-plot__btn python-plot__clear-output"
            onClick={onClearOutput}
            disabled={running || !hasClearableOutput}
            title="Clear this script's retained results and console without resetting variables"
          >
            Clear output
          </button>
        </div>

        <div
          id={`${outputId}-${activePane}-panel`}
          className="python-plot__output-panel"
          role="tabpanel"
          aria-labelledby={`${outputId}-${activePane}-tab`}
        >
          {activePane === "result" ? (
            <ResultPane result={runResult} onExport={onExportOutput} />
          ) : (
            <ConsolePane
              output={consoleOutput}
              live={ownsOutput ? liveStream : null}
              onJumpToLine={onJumpToLine}
            />
          )}
        </div>
      </div>
    </div>
  );
}

interface PyodideStatusBadgeProps {
  status: PyodideStatus;
  running: boolean;
}

function PyodideStatusBadge({ status, running }: PyodideStatusBadgeProps) {
  let label = "Idle";
  let tone: "idle" | "loading" | "ready" | "error" = "idle";
  if (status.kind === "loading-runtime") {
    label = "Booting Pyodide…";
    tone = "loading";
  } else if (status.kind === "loading-package") {
    label = `Loading ${status.name}…`;
    tone = "loading";
  } else if (status.kind === "ready") {
    label = running ? "Running" : "Ready";
    tone = running ? "loading" : "ready";
  } else if (status.kind === "error") {
    label = `Error: ${status.message}`;
    tone = "error";
  }
  return (
    <span
      className={`python-plot__status python-plot__status--${tone}`}
      role="status"
      aria-live="polite"
      aria-atomic="true"
    >
      <span className="python-plot__status-dot" aria-hidden="true" />
      <span>{label}</span>
    </span>
  );
}

/* ---------- Result rendering ---------- */

interface BuiltPlot {
  data: Data[];
  layout: Partial<Layout>;
  notice: string | null;
}

function buildArrayPlot(preview: ArrayLike<number> | null, shape: number[], viewport: PlaneViewportBounds | null, colorRange: readonly [number, number] | null): BuiltPlot | null {
  if (shape.length < 1 || shape.length > 2) return null;
  const display = preview ? preparePythonArrayDisplay(preview, shape, viewport ? {
    xRange: [viewport.xMin, viewport.xMax], yRange: [viewport.yMin, viewport.yMax],
  } : {}) : null;
  if (display?.kind === "line") {
    return {
      data: [
        {
          type: "scatter",
          mode: "lines",
          x: display.x,
          y: display.y,
          line: { color: "#4a8cff", width: 1.5 },
        },
      ],
      layout: {
        xaxis: { title: { text: "index" } },
        yaxis: { title: { text: "value" } },
        title: { text: "Python result (1D)", font: { size: 13 } },
      },
      notice: display.reduced
        ? `Renderer view keeps ${display.displayPoints.toLocaleString()} ordered endpoint/extrema/missing-data points `
          + `from ${display.sourcePoints.toLocaleString()}. The result and exports remain full resolution.`
        : null,
    };
  }
  if (display?.kind === "heatmap") {
    return {
      data: [
        {
          type: "heatmap",
          zsmooth: false,
          x: display.x,
          y: display.y,
          z: display.z,
          zmin: colorRange?.[0],
          zmax: colorRange?.[1],
          colorscale: "Inferno",
          opacity: 1.0,
          colorbar: { title: { text: "value", side: "right" } },
        },
      ],
      layout: {
        xaxis: { title: { text: "col" } },
        yaxis: { title: { text: "row" } },
        title: { text: "Python result (2D)", font: { size: 13 } },
      },
      notice: display.reduced
        ? `Renderer view point-samples ${display.displayShape[0].toLocaleString()}×`
          + `${display.displayShape[1].toLocaleString()} of ${display.sourceShape[0].toLocaleString()}×`
          + `${display.sourceShape[1].toLocaleString()} cells. Zooming refines the visible samples. `
          + "The result and exports remain full resolution; use Matplotlib for a chosen aggregation."
        : null,
    };
  }
  return null;
}

/** Single output. HTML renders in a sandboxed iframe to contain third-party
 *  CSS and scripts (xarray repr, matplotlib animation player). */
const OutputView = memo(function OutputView({ output, onExport }: {
  output: PythonOutput;
  onExport: (output: PythonOutput, format?: ArrayExportFormat) => void;
}) {
  const label = output.label ? (
    <div className="python-plot__output-label">{output.label}</div>
  ) : null;

  if (output.kind === "image") {
    return (
      <div className="python-plot__output">
        {label}
        <img
          alt={output.label ?? "figure"}
          src={output.svg ? `data:image/svg+xml;charset=utf-8,${encodeURIComponent(output.svg)}` : `data:image/png;base64,${output.pngBase64}`}
          style={output.svg ? { width: "100%" } : undefined}
          onError={(event) => {
            if (event.currentTarget.src.startsWith("data:image/svg+xml")) event.currentTarget.src = `data:image/png;base64,${output.pngBase64}`;
          }}
          className="python-plot__fig"
        />
        <div className="python-plot__output-actions">
          <button type="button" className="python-plot__btn" onClick={() => onExport(output)}>
            Save PNG
          </button>
          {output.svg && <button type="button" className="python-plot__btn" onClick={() => onExport(output, "svg")}>Save SVG</button>}
        </div>
      </div>
    );
  }

  if (output.kind === "animation") {
    return (
      <div className="python-plot__output">
        {label}
        <AnimationView output={output} onExport={onExport} />
      </div>
    );
  }

  if (output.kind === "html") {
    return (
      <div className="python-plot__output">
        {label}
        <iframe
          className="python-plot__html"
          title={output.label ?? "Python output"}
          sandbox={output.scripts ? "allow-scripts" : ""}
          srcDoc={output.html}
        />
      </div>
    );
  }

  if (output.kind === "ndarray") {
    return <PythonArrayOutput output={output} onExport={onExport} />;
  }

  return (
    <div className="python-plot__output">
      {label}
      <pre className="python-plot__console-stdout">{output.text}</pre>
    </div>
  );
});

const pythonArrayIdentities = new WeakMap<object, string>();
let pythonArrayIdentitySequence = 0;
function pythonArrayIdentity(data: PythonNumericArray): string {
  let identity = pythonArrayIdentities.get(data);
  if (!identity) { identity = `python-array-${++pythonArrayIdentitySequence}`; pythonArrayIdentities.set(data, identity); }
  return identity;
}

const PythonArrayOutput = memo(function PythonArrayOutput({ output, onExport }: {
  output: Extract<PythonOutput, { kind: "ndarray" }>;
  onExport: (output: PythonOutput, format?: ArrayExportFormat) => void;
}) {
  const identity = pythonArrayIdentity(output.data);
  const { viewport, onRelayout } = usePlaneViewport(identity, 120);
  const pendingViewport = useRef<{ identity: string; bounds: PlaneViewportBounds | null }>({ identity, bounds: null });
  const preview = useMemo(() => pythonArrayPreview(output.data), [output.data]);
  const colorRange = useMemo((): [number, number] | null => {
    if (!preview) return null;
    let minimum = Infinity, maximum = -Infinity;
    for (let index = 0; index < preview.length; index++) {
      const value = preview[index];
      if (Number.isFinite(value)) { minimum = Math.min(minimum, value); maximum = Math.max(maximum, value); }
    }
    if (!Number.isFinite(minimum)) return null;
    if (minimum === maximum) { const pad = Math.max(1, Math.abs(minimum) * 0.01); return [minimum - pad, maximum + pad]; }
    return [minimum, maximum];
  }, [preview]);
  const plot = useMemo(() => buildArrayPlot(preview, output.shape, viewport, colorRange), [preview, output.shape, viewport, colorRange]);
  const relayout = useCallback((event: Readonly<Record<string, unknown>>) => {
    const next = pythonArrayViewportFromRelayout(event,
      pendingViewport.current.identity === identity ? pendingViewport.current.bounds : null, output.shape, colorRange);
    if (next === undefined) return;
    pendingViewport.current = { identity, bounds: next };
    onRelayout(next ? { "xaxis.range": [next.xMin, next.xMax], "yaxis.range": [next.yMin, next.yMax] }
      : { "xaxis.autorange": true, "yaxis.autorange": true });
  }, [identity, output.shape, colorRange, onRelayout]);
  return <div className="python-plot__output">
    {output.label && <div className="python-plot__output-label">{output.label}</div>}
    {plot ? <><PlotCanvas data={plot.data} layout={plot.layout} viewIdentity={identity} onRelayout={relayout} />
      {plot.notice && <div className="python-plot__display-note">{plot.notice}</div>}</>
      : <div className="python-plot__empty">{output.shape.join("×")} array ({output.dtype ?? pythonArrayDtype(output.data)}) — plotting requires 1D/2D real values within JavaScript’s exact integer range; reduce or slice to 1D/2D, or transform explicitly (e.g. <code>field[:, :, 0]</code>), or <code>display(animate(...))</code> over an axis.</div>}
    <ArrayExportActions output={output} onExport={onExport} />
  </div>;
});

/** Which file an array leaves as. */
type ArrayExportFormat = "npy" | "csv" | "svg";

/** Array export: .npy always available (preserves dtype and shape);
 *  CSV only for rank ≤2 and below MAX_CSV_CELLS. Button stays visible
 *  when disabled so the limit is discoverable. */
function ArrayExportActions({ output, onExport }: {
  output: Extract<PythonOutput, { kind: "ndarray" }>;
  onExport: (output: PythonOutput, format?: ArrayExportFormat) => void;
}) {
  const cells = output.shape.reduce((total, n) => total * n, 1);
  const tabular = output.shape.length >= 1 && output.shape.length <= 2;
  const csvTooBig = cells > MAX_CSV_CELLS;
  return (
    <div className="python-plot__output-actions">
      <button
        type="button"
        className="python-plot__btn"
        onClick={() => onExport(output, "npy")}
        title={`Save ${output.shape.join("×")} ${output.dtype ?? pythonArrayDtype(output.data)} as .npy — np.load reads it back unchanged`}
      >
        Save .npy
      </button>
      {tabular && (
        <button
          type="button"
          className="python-plot__btn"
          onClick={() => onExport(output, "csv")}
          disabled={csvTooBig}
          title={csvTooBig
            ? `${cells.toLocaleString()} values is past the ${MAX_CSV_CELLS.toLocaleString()}-cell `
              + `CSV limit; .npy has no limit and is smaller`
            : "Save as comma-separated text, C order"}
        >
          Save CSV
        </button>
      )}
    </div>
  );
}

/** Frame player. Frames are pre-rendered PNGs, so scrubbing costs nothing and
 *  export needs no re-run of the user's code. */
function AnimationView({ output, onExport }: {
  output: Extract<PythonOutput, { kind: "animation" }>;
  onExport: (output: PythonOutput) => void;
}) {
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const count = output.framesBase64.length;

  useEffect(() => {
    if (!playing || count <= 1) return;
    const timer = window.setInterval(
      () => setFrame((current) => (current + 1) % count),
      Math.max(16, Math.round(1000 / output.fps)),
    );
    return () => window.clearInterval(timer);
  }, [playing, count, output.fps]);

  // A shorter re-run must not leave the index past the end of the new sequence.
  const safeFrame = count === 0 ? 0 : Math.min(frame, count - 1);
  if (count === 0) return <div className="python-plot__empty">Animation has no frames.</div>;

  return (
    <>
      <img
        alt={`frame ${safeFrame + 1} of ${count}`}
        src={`data:image/png;base64,${output.framesBase64[safeFrame]}`}
        className="python-plot__fig"
      />
      <div className="python-plot__output-actions">
        <button
          type="button"
          className="python-plot__btn"
          onClick={() => setPlaying((value) => !value)}
        >
          <Icon name={playing ? "pause" : "play"} size={10} />
          <span>{playing ? "Pause" : "Play"}</span>
        </button>
        <input
          type="range"
          min={0}
          max={count - 1}
          value={safeFrame}
          onChange={(event) => { setPlaying(false); setFrame(Number(event.target.value)); }}
          aria-label="Animation frame"
        />
        <span className="python-plot__refs">
          {safeFrame + 1}/{count} @ {output.fps} fps
        </span>
        <button type="button" className="python-plot__btn" onClick={() => onExport(output)}>
          Export…
        </button>
      </div>
    </>
  );
}

interface ResultPaneProps {
  result: RunResult | null;
  onExport: (output: PythonOutput, format?: ArrayExportFormat) => void;
}

const ResultPane = memo(function ResultPane({ result, onExport }: ResultPaneProps) {
  if (!result) {
    return <div className="python-plot__empty">Hit Run to evaluate the cell.</div>;
  }
  if (result.outputs.length === 0) {
    return (
      <div className="python-plot__empty">
        Cell ran without showing anything. Assign <code>result</code>, draw with
        matplotlib, or call <code>display(...)</code>.
      </div>
    );
  }
  return (
    <div className="python-plot__result python-plot__result--stack">
      {result.outputs.map((output, index) => (
        <OutputView key={index} output={output} onExport={onExport} />
      ))}
    </div>
  );
});

/** stderr with user frames turned into jump-to-line buttons.
 *  Only File "<exec>" frames are linked; stdlib/Pyodide frames are skipped. */
function TracebackText({
  text,
  onJumpToLine,
}: {
  text: string;
  onJumpToLine: ((line: number) => void) | undefined;
}) {
  if (!onJumpToLine) return <>{text}</>;
  return (
    <>
      {splitTraceback(text).map((segment, index) =>
        segment.line === null ? (
          <span key={index}>{segment.text}</span>
        ) : (
          <button
            key={index}
            type="button"
            className="python-plot__tb-link"
            onClick={() => onJumpToLine(segment.line!)}
            title={`Go to line ${segment.line}`}
          >
            {segment.text}
          </button>
        ),
      )}
    </>
  );
}

function ConsolePane({
  output,
  live,
  onJumpToLine,
}: {
  output: { stdout: string; stderr: string };
  live: LiveStream | null;
  onJumpToLine?: (line: number) => void;
}) {
  if (live) {
    const streaming = live.stdout.length > 0 || live.stderr.length > 0;
    return (
      <div className="python-plot__console">
        <div className="python-plot__console-live">
          {streaming ? "Running — output so far" : "Running…"}
          {live.truncated && ` · showing the last ${MAX_LIVE_STREAM_CHARS.toLocaleString()} characters`}
        </div>
        {live.stdout.length > 0 && (
          <pre className="python-plot__console-stdout">{live.stdout}</pre>
        )}
        {live.stderr.length > 0 && (
          <pre className="python-plot__console-stderr">
          <TracebackText text={live.stderr} onJumpToLine={onJumpToLine} />
        </pre>
        )}
      </div>
    );
  }
  const hasStdout = output.stdout.length > 0;
  const hasStderr = output.stderr.length > 0;
  if (!hasStdout && !hasStderr) {
    return <div className="python-plot__empty">No output yet.</div>;
  }
  return (
    <div className="python-plot__console">
      {hasStdout && (
        <pre className="python-plot__console-stdout">{output.stdout}</pre>
      )}
      {hasStderr && (
        <pre className="python-plot__console-stderr">
          <TracebackText text={output.stderr} onJumpToLine={onJumpToLine} />
        </pre>
      )}
    </div>
  );
}

/* ---------- Gallery sidebar ---------- */

interface PluginGalleryProps {
  plugins: PluginEntry[];
  dir: string | null;
  loading: boolean;
  error: string | null;
  activePath: string | null;
  onClose: () => void;
  onRefresh: () => Promise<void>;
  onOpenDir: () => Promise<void>;
  onPick: (plugin: PluginEntry) => Promise<void>;
}

function PluginGallery({
  plugins,
  dir,
  loading,
  error,
  activePath,
  onClose,
  onRefresh,
  onOpenDir,
  onPick,
}: PluginGalleryProps) {
  return (
    <aside className="plugin-gallery">
      <header className="plugin-gallery__header">
        <span className="plugin-gallery__title">Plugins</span>
        <button
          type="button"
          className="python-plot__btn"
          onClick={() => void onRefresh()}
          title="Re-scan ~/.nimbus/plugins/"
          disabled={loading}
        >
          {loading ? "…" : "Refresh"}
        </button>
        <button
          type="button"
          className="python-plot__btn"
          onClick={() => void onOpenDir()}
          title="Reveal plugins folder in Finder"
        >
          Open dir
        </button>
        <button
          type="button"
          className="python-plot__btn"
          onClick={onClose}
          title="Hide gallery"
        >
          <Icon name="x" size={10} />
        </button>
      </header>
      {dir && (
        <div className="plugin-gallery__dir" title={dir}>
          {compactPath(dir)}
        </div>
      )}
      {error && (
        <div className="plugin-gallery__error" role="alert">
          <Icon name="warning" size={12} />
          <span>{error}</span>
        </div>
      )}
      {plugins.length === 0 ? (
        <div className="plugin-gallery__empty">
          {loading ? (
            <span>Scanning…</span>
          ) : (
            <>
              <span>No plugins yet.</span>
              <span className="plugin-gallery__empty-hint">
                Save a snippet to populate the gallery.
              </span>
            </>
          )}
        </div>
      ) : (
        <ul className="plugin-gallery__list">
          {plugins.map((p) => (
            /* Button inside li, not a clickable li: proper role, tab stop, Enter. */
            <li key={p.path}>
              <button
                type="button"
                className={`plugin-gallery__item ${p.path === activePath ? "plugin-gallery__item--active" : ""}`}
                onClick={() => void onPick(p)}
                title={p.path}
                aria-current={p.path === activePath ? "true" : undefined}
              >
                <span className="plugin-gallery__item-name">{p.name}</span>
                <span className="plugin-gallery__item-meta">
                  {formatSize(p.sizeBytes)} · {formatRelativeTime(p.modifiedMs)}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}

function compactPath(path: string): string {
  // Tighten the dir display: /Users/<name>/.nimbus/plugins → ~/.nimbus/plugins.
  // Linux $HOME (/home/<name>) gets the same treatment.
  for (const prefix of ["/Users/", "/home/"]) {
    if (path.startsWith(prefix)) {
      const home = path.split("/").slice(0, 3).join("/");
      if (path.startsWith(home)) return `~${path.slice(home.length)}`;
    }
  }
  return path;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatRelativeTime(ms: number | null): string {
  if (ms == null) return "—";
  const dt = Date.now() - ms;
  if (dt < 60 * 1000) return "just now";
  if (dt < 60 * 60 * 1000) return `${Math.floor(dt / (60 * 1000))}m ago`;
  if (dt < 24 * 60 * 60 * 1000) return `${Math.floor(dt / (60 * 60 * 1000))}h ago`;
  return `${Math.floor(dt / (24 * 60 * 60 * 1000))}d ago`;
}

/* ---------- helpers ---------- */
